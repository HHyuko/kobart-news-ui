
from flask import Flask, request, jsonify, render_template
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

app = Flask(__name__)

# KoBART 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

def search_naver_news(keyword):
    base_url = "https://search.naver.com/search.naver"
    params = {
        "where": "news",
        "query": keyword,
        "sm": "tab_opt",
        "sort": 0
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
    }

    response = requests.get(base_url, params=params, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a_tag in soup.select("a.info"):
        href = a_tag.get("href", "")
        if "news.naver.com" in href:
            links.append(href)
        if len(links) >= 20:
            break

    return links

def extract_article_info(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.select_one(".media_end_head_headline")
        press_tag = soup.select_one(".media_end_head_top_logo img")
        date_tag = soup.select_one(".media_end_head_info_datestamp_time")
        content_tag = soup.select_one("#dic_area")

        title = title_tag.get_text(strip=True) if title_tag else "제목 없음"
        press = press_tag['alt'] if press_tag and 'alt' in press_tag.attrs else "언론사 없음"
        date = date_tag['data-date-time'][:10] if date_tag and 'data-date-time' in date_tag.attrs else "날짜 없음"
        content = content_tag.get_text(strip=True) if content_tag else ""

        return {
            "title": title,
            "press": press,
            "date": date,
            "content": content,
            "url": url
        }

    except:
        return None

def is_similar(text1, text2, threshold=0.85):
    return SequenceMatcher(None, text1, text2).ratio() > threshold

def kobart_summarize(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=160, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    keyword = data.get('keyword', '')
    urls = search_naver_news(keyword)

    raw_articles = []
    for url in urls:
        info = extract_article_info(url)
        if info and len(info['content']) >= 50:
            raw_articles.append(info)

    deduped_articles = []
    for article in raw_articles:
        if all(not is_similar(article['title'], a['title']) for a in deduped_articles):
            deduped_articles.append(article)
        if len(deduped_articles) >= 10:
            break

    summaries = []
    full_summary_text = ""
    for info in deduped_articles:
        summary = kobart_summarize(info["content"])
        summaries.append(f"- {summary}")
        info["summary"] = summary
        full_summary_text += summary + " "

    newsletter = kobart_summarize(full_summary_text) if summaries else "관련 뉴스를 찾을 수 없거나 본문을 불러오지 못했습니다."

    output = f"🧠 {newsletter}\n\n"
    for info in deduped_articles:
        output += f"📰 {info['title']}\n📎 {info['url']}\n🗞️ {info['press']} | {info['date']}\n📝 {info['summary']}\n\n"

    return jsonify({'summary': output.strip()})

if __name__ == '__main__':
    app.run(debug=True)
