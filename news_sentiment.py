import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import requests
import feedparser
from collections import defaultdict
import re
import json

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)

# API Keys
NEWS_API_KEY = "HeFlh3NwNfVngiiaWWtLmayOjUDdgZWmNwk1sI2g"
DART_API_KEY = "e45fa610cea4a8e8a6eebd9e05e3580daa071f82"

# 한국 주요 언론사 RSS 피드
RSS_FEEDS = {
    "한국경제": "https://www.hankyung.com/feed/finance",
    "매일경제": "https://www.mk.co.kr/rss/50100032",
    "조선비즈": "https://biz.chosun.com/site/data/rss/rss.xml",
    "이데일리": "https://rss.edaily.co.kr/stock_news.xml",
    "연합인포맥스": "https://www.yna.co.kr/rss/economy.xml"
}

# 주식 종목 매핑 (종목명과 코드)
STOCK_MAPPING = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "카카오": "035720",
    "현대차": "005380",
    "네이버": "035420",
    "NAVER": "035420",
    "LG화학": "051910",
    "삼성SDI": "006400",
    "삼성물산": "028260",
    "KB금융": "105560",
    "신한지주": "055550",
    "현대모비스": "012330",
    "기아": "000270",
    "포스코": "005490",
    "셀트리온": "068270",
    "삼성바이오로직스": "207940"
}

class EnhancedNewsAnalyzer:
    """향상된 뉴스 분석 엔진"""
    
    def __init__(self):
        self.positive_keywords = {
            'strong': ['최고치', '신고가', '급등', '상승', '호재', '실적개선', '흑자전환', '수주', '계약'],
            'medium': ['상승세', '증가', '개선', '성장', '확대', '긍정적', '호조', '회복'],
            'weak': ['소폭상승', '보합', '안정', '유지']
        }
        
        self.negative_keywords = {
            'strong': ['폭락', '급락', '적자', '파산', '리콜', '소송', '제재', '스캔들'],
            'medium': ['하락', '감소', '부진', '우려', '불안', '약세', '조정'],
            'weak': ['소폭하락', '보합', '횡보']
        }
        
        self.financial_terms = ['매출', '영업이익', '순이익', 'PER', 'PBR', 'ROE', '부채비율']
    
    def analyze_sentiment_korean(self, text: str) -> Tuple[str, float]:
        """한국어 텍스트 감성 분석"""
        text_lower = text.lower()
        
        # 점수 계산
        positive_score = 0
        negative_score = 0
        
        # 긍정 키워드 점수
        for strength, keywords in self.positive_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if strength == 'strong':
                        positive_score += 3
                    elif strength == 'medium':
                        positive_score += 2
                    else:
                        positive_score += 1
        
        # 부정 키워드 점수
        for strength, keywords in self.negative_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if strength == 'strong':
                        negative_score += 3
                    elif strength == 'medium':
                        negative_score += 2
                    else:
                        negative_score += 1
        
        # 최종 점수 계산
        total_score = positive_score - negative_score
        
        # 감성 및 신뢰도 결정
        if total_score > 3:
            sentiment = "positive"
            confidence = min(0.9, 0.6 + total_score * 0.05)
        elif total_score < -3:
            sentiment = "negative"
            confidence = min(0.9, 0.6 + abs(total_score) * 0.05)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return sentiment, confidence
    
    def extract_stock_mentions(self, text: str) -> List[Tuple[str, str]]:
        """텍스트에서 종목 추출"""
        mentioned_stocks = []
        
        for stock_name, stock_code in STOCK_MAPPING.items():
            if stock_name in text:
                mentioned_stocks.append((stock_name, stock_code))
        
        return mentioned_stocks


def get_news_from_api(days_back: int = 1) -> List[Dict]:
    """News API에서 뉴스 가져오기"""
    articles = []
    
    try:
        # 한국 경제 뉴스 검색
        params = {
            "q": "주식 OR 증시 OR 코스피 OR 코스닥",
            "from": (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
            "language": "ko",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": NEWS_API_KEY,
        }
        
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        analyzer = EnhancedNewsAnalyzer()
        
        for art in data.get("articles", []):
            title = art.get("title", "")
            description = art.get("description", "")
            url = art.get("url", "")
            published_at = art.get("publishedAt", "")
            
            if not title or not published_at:
                continue
            
            try:
                date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except ValueError:
                date = datetime.utcnow()
            
            # 감성 분석
            full_text = f"{title} {description}"
            sentiment, confidence = analyzer.analyze_sentiment_korean(full_text)
            
            # 종목 추출
            mentioned_stocks = analyzer.extract_stock_mentions(full_text)
            
            articles.append({
                "title": title,
                "description": description,
                "date": date,
                "sentiment": sentiment,
                "confidence": confidence,
                "url": url,
                "source": art.get("source", {}).get("name", "Unknown"),
                "mentioned_stocks": mentioned_stocks
            })
        
    except Exception as e:
        logger.error(f"Failed to fetch from News API: {e}")
    
    return articles


def get_news_from_rss() -> List[Dict]:
    """RSS 피드에서 뉴스 가져오기"""
    articles = []
    analyzer = EnhancedNewsAnalyzer()
    
    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:10]:  # 각 소스에서 최근 10개만
                title = entry.get("title", "")
                description = entry.get("summary", "")
                link = entry.get("link", "")
                
                # 날짜 파싱
                published = entry.get("published_parsed")
                if published:
                    date = datetime.fromtimestamp(feedparser._parse_date(entry.published).timestamp())
                else:
                    date = datetime.utcnow()
                
                # 감성 분석
                full_text = f"{title} {description}"
                sentiment, confidence = analyzer.analyze_sentiment_korean(full_text)
                
                # 종목 추출
                mentioned_stocks = analyzer.extract_stock_mentions(full_text)
                
                articles.append({
                    "title": title,
                    "description": description[:200] + "..." if len(description) > 200 else description,
                    "date": date,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "url": link,
                    "source": source_name,
                    "mentioned_stocks": mentioned_stocks
                })
                
        except Exception as e:
            logger.error(f"Failed to fetch RSS from {source_name}: {e}")
    
    return articles


def get_dart_disclosures(stock_codes: List[str]) -> List[Dict]:
    """DART API에서 공시 정보 가져오기"""
    disclosures = []
    
    try:
        # 각 종목별로 공시 조회
        for stock_code in stock_codes:
            params = {
                'crtfc_key': DART_API_KEY,
                'corp_code': stock_code,  # 실제로는 corp_code 변환 필요
                'bgn_de': (datetime.now() - timedelta(days=7)).strftime('%Y%m%d'),
                'end_de': datetime.now().strftime('%Y%m%d'),
                'page_no': '1',
                'page_count': '10'
            }
            
            resp = requests.get(
                "https://opendart.fss.or.kr/api/list.json",
                params=params,
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                
                if data.get('status') == '000':
                    for item in data.get('list', []):
                        # 중요 공시 판단
                        report_nm = item.get('report_nm', '')
                        is_important = any(keyword in report_nm for keyword in 
                                         ['주요사항보고서', '분기보고서', '반기보고서', '사업보고서'])
                        
                        disclosures.append({
                            'stock_code': stock_code,
                            'corp_name': item.get('corp_name'),
                            'report_nm': report_nm,
                            'rcept_dt': item.get('rcept_dt'),
                            'is_important': is_important
                        })
                        
    except Exception as e:
        logger.error(f"Failed to fetch DART disclosures: {e}")
    
    return disclosures


def analyze_stock_sentiment(days_back: int = 1) -> Dict[str, Dict]:
    """종목별 감성 점수 종합 분석"""
    # 뉴스 수집
    all_news = []
    all_news.extend(get_news_from_api(days_back))
    all_news.extend(get_news_from_rss())
    
    # 종목별 감성 점수 집계
    stock_scores = defaultdict(lambda: {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'total_mentions': 0,
        'weighted_score': 0,
        'news_items': [],
        'important_news': []
    })
    
    for news in all_news:
        for stock_name, stock_code in news['mentioned_stocks']:
            stock_scores[stock_code]['total_mentions'] += 1
            stock_scores[stock_code]['news_items'].append(news)
            
            # 감성별 카운트
            sentiment = news['sentiment']
            confidence = news['confidence']
            
            if sentiment == 'positive':
                stock_scores[stock_code]['positive'] += 1
                stock_scores[stock_code]['weighted_score'] += confidence
            elif sentiment == 'negative':
                stock_scores[stock_code]['negative'] += 1
                stock_scores[stock_code]['weighted_score'] -= confidence
            else:
                stock_scores[stock_code]['neutral'] += 1
            
            # 중요 뉴스 판단 (신뢰도 높고 최근)
            if confidence > 0.7 and news['date'] > datetime.now() - timedelta(hours=12):
                stock_scores[stock_code]['important_news'].append(news)
    
    # 최종 점수 계산 및 추천 결정
    for stock_code, scores in stock_scores.items():
        total = scores['total_mentions']
        if total > 0:
            # 종합 점수 계산 (언급 횟수도 고려)
            scores['final_score'] = (scores['weighted_score'] / total) * min(1 + total/10, 2)
            
            # 추천 유형 결정
            if scores['final_score'] > 0.3 and scores['positive'] > scores['negative']:
                scores['recommendation'] = 'BUY'
            elif scores['final_score'] < -0.3 and scores['negative'] > scores['positive']:
                scores['recommendation'] = 'SELL'
            else:
                scores['recommendation'] = 'HOLD'
    
    return dict(stock_scores)


def get_recent_news(days_back: int = 1) -> List[Dict]:
    """통합 뉴스 수집 (기존 함수 호환성 유지)"""
    all_news = []
    all_news.extend(get_news_from_api(days_back))
    all_news.extend(get_news_from_rss())
    
    # 시간순 정렬
    all_news.sort(key=lambda x: x['date'], reverse=True)
    
    # 중복 제거 (제목 기반)
    seen_titles = set()
    unique_news = []
    
    for news in all_news:
        title_key = news['title'][:50]  # 제목 앞 50자로 중복 체크
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_news.append(news)
    
    return unique_news[:50]  # 최대 50개 반환
