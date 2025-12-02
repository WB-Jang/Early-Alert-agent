import os
import json
from typing import TypedDict, List, Annotated
from newsapi import NewsApiClient
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. Configuration & Model
# ==========================================
# .env에서 로드하거나 직접 입력
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_KEY_HERE")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/qwen2.5-7b-instruct-q4_k_m.gguf")

# 모델 로드 (전역 인스턴스)
# n_gpu_layers=-1 : 모든 레이어를 GPU에 할당 (RTX 4060 필수)
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, 
    n_ctx=4096,
    temperature=0.1,
    max_tokens=1024,
    verbose=False
)

# ==========================================
# 2. Define State (Graph State)
# ==========================================
class AgentState(TypedDict):
    raw_headlines: List[dict]  # API에서 가져온 원본 뉴스
    relevant_news: List[dict]  # 1차 필터링 통과한 뉴스
    analyzed_risks: List[dict] # 2차 분석(리스크 평가) 완료된 뉴스
    final_email_body: str      # 최종 발송할 이메일 본문

# ==========================================
# 3. Define Nodes
# ==========================================

def fetch_news_node(state: AgentState):
    """뉴스 API 호출 노드"""
    print("--- [Step 1] Fetching News ---")
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        # 실제 환경에선 'finance', 'economy', 'banking' 등 쿼리 사용
        response = newsapi.get_top_headlines(q='economy', language='en', page_size=10)
        articles = response.get('articles', [])
    except Exception as e:
        print(f"API Error: {e}, using Mock Data")
        # Mock Data for Testing
        articles = [
            {"title": "Central Bank hikes rates by 0.75%", "description": "Inflation control measures...", "url": "http://test.com/1"},
            {"title": "New K-Pop star debut", "description": "Entertainment news...", "url": "http://test.com/2"}
        ]
    
    return {"raw_headlines": articles}

def filter_news_node(state: AgentState):
    """1차 필터링 노드 (배치 처리가 아닌 순차 처리 예시)"""
    print("--- [Step 2] Filtering News ---")
    raw_news = state['raw_headlines']
    relevant_news = []

    filter_prompt = PromptTemplate(
        template="""Check if this headline is related to 'Financial Risk', 'Banking', or 'Economy'.
        Headline: "{headline}"
        Return JSON: {{"is_relevant": true/false}}""",
        input_variables=["headline"]
    )

    for news in raw_news:
        try:
            res = llm.invoke(filter_prompt.format(headline=news['title']))
            # JSON 파싱 (단순화)
            if '"is_relevant": true' in res.lower():
                relevant_news.append(news)
                print(f"  -> Relevant: {news['title'][:30]}...")
            else:
                print(f"  -> Skipped: {news['title'][:30]}...")
        except:
            continue
            
    return {"relevant_news": relevant_news}

def analyze_risk_node(state: AgentState):
    """2차 심층 분석 노드"""
    print("--- [Step 3] Analyzing Risks ---")
    relevant_news = state['relevant_news']
    analyzed_risks = []

    analysis_prompt = PromptTemplate(
        template="""Analyze for Credit/Market Risk.
        Headline: "{headline}"
        Content: "{content}"
        
        Return JSON format:
        {{
            "risk_type": "Market Risk/Credit Risk/None",
            "impact": "High/Medium/Low",
            "summary": "Short summary in Korean"
        }}
        """,
        input_variables=["headline", "content"]
    )

    for news in relevant_news:
        content = news.get('description') or news['title']
        res = llm.invoke(analysis_prompt.format(headline=news['title'], content=content))
        
        # 실제 구현에선 견고한 JSON Parser 필요
        # 여기서는 LLM이 JSON 포맷을 잘 지킨다고 가정하고 텍스트 처리
        if "High" in res or "Medium" in res:
             analyzed_risks.append({
                 "original": news,
                 "analysis": res # 실제로는 여기서 파싱된 데이터를 넣어야 함
             })

    return {"analyzed_risks": analyzed_risks}

def report_node(state: AgentState):
    """이메일 작성 노드"""
    print("--- [Step 4] Writing Report ---")
    risks = state['analyzed_risks']
    
    if not risks:
        return {"final_email_body": "No significant risks detected."}

    # 최종 보고서 작성 프롬프트
    input_text = "\n".join([str(r['analysis']) for r in risks])
    
    writer_prompt = f"""
    Based on the following risk analyses, write a cohesive email alert in Korean.
    Format clearly with bullet points.
    
    Analyses:
    {input_text}
    """
    
    email_body = llm.invoke(writer_prompt)
    
    # 여기서 실제 이메일 발송 로직 수행 (SMTP)
    print("\n========== [FINAL EMAIL] ==========")
    print(email_body)
    print("===================================")

    return {"final_email_body": email_body}

# ==========================================
# 4. Build Graph
# ==========================================

workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("fetch", fetch_news_node)
workflow.add_node("filter", filter_news_node)
workflow.add_node("analyze", analyze_risk_node)
workflow.add_node("report", report_node)

# 엣지 연결 (Linear Flow)
workflow.set_entry_point("fetch")
workflow.add_edge("fetch", "filter")

# 조건부 엣지: 필터된 뉴스가 없으면 바로 종료
def check_relevance(state: AgentState):
    if not state['relevant_news']:
        print("--- No relevant news found. Ending. ---")
        return "end"
    return "analyze"

workflow.add_conditional_edges(
    "filter",
    check_relevance,
    {
        "analyze": "analyze",
        "end": END
    }
)

workflow.add_edge("analyze", "report")
workflow.add_edge("report", END)

# 컴파일
app = workflow.compile()

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    # LangGraph 실행
    app.invoke({"raw_headlines": []})
