import os
import json
import time
from datetime import datetime
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from newsapi import NewsApiClient # 뉴스 수집용 (가입 필요, 없으면 Mock 데이터 사용)

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
MODEL_PATH = os.getenv("MODEL_PATH", "./models/qwen2.5-7b-instruct-q4_k_m.gguf")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY") # https://newsapi.org/ 에서 무료 키 발급 가능

# 하드웨어 설정 (RTX 4060 8GB 최적화)
n_gpu_layers = -1  # 모든 레이어를 GPU에 할당
n_ctx = 4096       # 컨텍스트 윈도우

# ==========================================
# 2. 모델 로드 (Initialize LLM)
# ==========================================
print(">>> Loading AI Model... (This may take a moment)")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_ctx=n_ctx,
    temperature=0.1, # 분석용이므로 낮은 온도로 설정 (Fact 위주)
    max_tokens=512,
    verbose=False
)
print(">>> Model Loaded Successfully.")

# ==========================================
# 3. 프롬프트 정의 (Prompts)
# ==========================================

# Step 1: 1차 필터링 (관련성 체크)
filter_template = """
You are a Financial News Filter. Analyze the headline below.
Determine if it is potentially related to 'Credit Risk', 'Market Risk', 'Macroeconomics', or 'Banking'.
Ignore sports, entertainment, and general crimes.

Headline: "{headline}"

Return ONLY a JSON object strictly in this format:
{{"is_relevant": true, "reason": "reason"}}
or
{{"is_relevant": false, "reason": "reason"}}
"""
filter_prompt = PromptTemplate(template=filter_template, input_variables=["headline"])

# Step 2: 2차 심층 분석 (리스크 평가)
analysis_template = """
You are a Senior Risk Analyst. Analyze this news for potential impact on a commercial bank.

Headline: "{headline}"
Content Snippet: "{content}"

Analyze step-by-step:
1. Is this a Market Risk (interest rates, FX, stocks)?
2. Is this a Credit Risk (bankruptcy, debt crisis)?
3. Is the impact High or Medium?

Return ONLY a JSON object strictly in this format:
{{
  "risk_type": "Market Risk" or "Credit Risk" or "None",
  "impact_level": "High" or "Medium" or "Low",
  "send_alert": true or false
}}
(Set 'send_alert' to true ONLY if impact is High or Medium)
"""
analysis_prompt = PromptTemplate(template=analysis_template, input_variables=["headline", "content"])

# Step 3: 요약 및 번역 (최종 아웃풋)
summary_template = """
You are an AI Executive Assistant.
Summarize the following financial news into a Korean briefing format.

Headline: "{headline}"
Content: "{content}"

Output strictly in Korean(한국어) in this format:

**[긴급] {risk_type} 조기 경보**
* **헤드라인:** (Korean Translation)
* **핵심 요약:**
  - (Point 1)
  - (Point 2)
* **리스크 요인:** (One sentence summary of the threat)
"""
summary_prompt = PromptTemplate(template=summary_template, input_variables=["headline", "content", "risk_type"])

# ==========================================
# 4. 기능 함수 구현 (Functions)
# ==========================================

def fetch_news():
    """
    뉴스 API를 통해 최신 뉴스를 가져옵니다.
    API 키가 없거나 에러 발생 시 테스트용 Mock 데이터를 반환합니다.
    """
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        # 키워드: bank, economy, credit, market, finance
        # 언어: en (영어), 실제 구현시에는 여러 언어 쿼리 필요
        top_headlines = newsapi.get_top_headlines(q='economy', language='en', page_size=5)
        articles = top_headlines.get('articles', [])
        if articles:
            return articles
    except Exception as e:
        print(f"[Info] API Call Failed or No Key. Using Mock Data. ({e})")
    
    # 테스트용 가짜 데이터 (Mock Data)
    return [
        {
            "title": "Central Bank announces surprise 0.5% interest rate hike due to inflation fears",
            "description": "The monetary policy committee decided to raise rates immediately. Markets are tumbling.",
            "url": "http://test-news.com/1",
            "source": {"name": "Global Finance"}
        },
        {
            "title": "New iPhone 16 features leaked ahead of launch",
            "description": "Apple's new phone will feature a better camera and AI capabilities.",
            "url": "http://test-news.com/2",
            "source": {"name": "Tech Daily"}
        },
        {
            "title": "Major Real Estate Developer files for bankruptcy protection",
            "description": "One of the largest developers has defaulted on its $5B debt obligations.",
            "url": "http://test-news.com/3",
            "source": {"name": "Biz Insider"}
        }
    ]

def parse_json_response(response_text):
    """LLM의 응답에서 JSON 부분만 추출하여 파싱합니다."""
    try:
        # Markdown 코드 블록 제거 등 전처리
        cleaned = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def send_email_alert(subject, body, to_email="user@example.com"):
    """
    실제 이메일 발송 함수 (여기서는 출력으로 대체)
    """
    print("\n" + "="*40)
    print(f"📧 [EMAIL SENT] To: {to_email}")
    print(f"Subject: {subject}")
    print("-" * 20)
    print(body)
    print("="*40 + "\n")
    # 실제 SMTP 구현 시 smtplib 사용

# ==========================================
# 5. 메인 실행 로직 (Main Pipeline)
# ==========================================

def run_early_warning_system():
    print(f"\n>>> Starting Scan at {datetime.now()}")
    articles = fetch_news()
    
    for article in articles:
        headline = article.get('title')
        content = article.get('description') or headline
        url = article.get('url')
        
        print(f"\nProcessing: {headline[:50]}...")
        
        # --- Step 1: 1차 필터링 ---
        try:
            filter_res_raw = llm.invoke(filter_prompt.format(headline=headline))
            filter_data = parse_json_response(filter_res_raw)
            
            if not filter_data or not filter_data.get('is_relevant'):
                print(f"   -> [Skipped] Irrelevant ({filter_data.get('reason') if filter_data else 'Parse Error'})")
                continue
        except Exception as e:
            print(f"   -> [Error] Filter Step: {e}")
            continue

        print("   -> [Relevant] Proceeding to Deep Analysis...")

        # --- Step 2: 2차 심층 분석 ---
        try:
            analysis_res_raw = llm.invoke(analysis_prompt.format(headline=headline, content=content))
            risk_data = parse_json_response(analysis_res_raw)
            
            if not risk_data or not risk_data.get('send_alert'):
                print(f"   -> [Safe] Low Risk or None ({risk_data.get('risk_type') if risk_data else 'Parse Error'})")
                continue
            
            risk_type = risk_data.get('risk_type')
            impact_level = risk_data.get('impact_level')
            print(f"   -> [ALERT] {impact_level} Impact {risk_type} Detected!")
            
        except Exception as e:
            print(f"   -> [Error] Analysis Step: {e}")
            continue

        # --- Step 3: 요약 및 발송 ---
        try:
            summary_res = llm.invoke(summary_prompt.format(
                headline=headline, 
                content=content,
                risk_type=risk_type
            ))
            
            # 이메일 본문 완성
            final_email_body = f"{summary_res}\n\n[Original Source]: {url}"
            send_email_alert(f"[Risk Alert] {risk_type} Detected", final_email_body)
            
        except Exception as e:
            print(f"   -> [Error] Summary Step: {e}")

if __name__ == "__main__":
    # 테스트를 위해 1회 실행
    run_early_warning_system()
    
    # 주기적 실행을 원하면 아래 주석 해제 (ex: 10분마다)
    # import schedule
    # schedule.every(10).minutes.do(run_early_warning_system)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
