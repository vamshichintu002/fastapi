from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from datetime import datetime
import json
import httpx
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize APIs with error checking
def init_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("Warning: OPENAI_API_KEY not found in environment variables")
        return None
    logger.info(f"Initializing OpenAI client with key: {api_key[:10]}...")
    return OpenAI(api_key=api_key)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    logger.warning("Warning: PERPLEXITY_API_KEY not found in environment variables")
else:
    logger.info(f"Loaded Perplexity API key: {PERPLEXITY_API_KEY[:10]}...")

client = init_openai()

# Model definitions
class FinancialGoal(BaseModel):
    goal_type: str
    target_amount: float
    timeline_years: int

class UserProfile(BaseModel):
    age: int
    employment_status: str
    annual_income: float
    marital_status: str
    dependents: int
    financial_goals: List[FinancialGoal]
    investment_horizon: str
    risk_tolerance: str
    comfort_with_fluctuations: int
    monthly_income: float
    monthly_expenses: float
    existing_debts: Optional[str]
    emergency_fund_months: Optional[int]
    investment_preferences: List[str]
    ethical_criteria: Optional[str]
    tax_advantaged_options: bool
    liquidity_needs: str
    investment_knowledge: str
    previous_investments: Optional[str]
    involvement_level: str

class InvestmentSuggestion(BaseModel):
    investment_type: str
    allocation_percentage: float
    details: str
    specific_suggestions: List[Dict[str, Any]]
    entry_strategy: str
    exit_strategy: str
    risk_mitigation: str

class PortfolioRecommendation(BaseModel):
    explanation: str
    recommendations: List[InvestmentSuggestion]
    market_analysis: Dict[str, Dict[str, Any]]
    review_schedule: str
    disclaimer: str

def get_fallback_market_analysis(asset_type: str) -> Dict:
    """Provide fallback market analysis when APIs fail"""
    market_trends = {
        "Stocks": {
            "current_trend": "Mixed trends in equity markets",
            "outlook": "Cautiously optimistic outlook",
            "key_factors": [
                "Global market conditions",
                "Domestic economic growth",
                "Corporate earnings"
            ],
            "risks": [
                "Market volatility",
                "Economic uncertainty",
                "Global factors"
            ],
            "specific_suggestions": [
                {
                    "name": "Large Cap Stock Fund",
                    "ticker": "NIFTY50",
                    "rationale": "Based on index performance"
                },
                {
                    "name": "Blue-chip Companies",
                    "ticker": "Various",
                    "rationale": "Stable, established companies"
                }
            ]
        },
        "Mutual_Funds": {
            "current_trend": "Steady growth in mutual fund investments",
            "outlook": "Positive for long-term investors",
            "key_factors": [
                "Professional management",
                "Diversification benefits"
            ],
            "risks": [
                "Market-linked returns",
                "Fund manager risk"
            ],
            "specific_suggestions": [
                {
                    "name": "Index Funds",
                    "ticker": "Various",
                    "rationale": "Low-cost market exposure"
                },
                {
                    "name": "Balanced Funds",
                    "ticker": "Various",
                    "rationale": "Mix of stocks and bonds"
                }
            ]
        },
        "Bonds": {
            "current_trend": "Stable returns in fixed income",
            "outlook": "Moderate yield expectations",
            "key_factors": [
                "Interest rate environment",
                "Credit quality"
            ],
            "risks": [
                "Interest rate risk",
                "Credit risk"
            ],
            "specific_suggestions": [
                {
                    "name": "Government Securities",
                    "ticker": "Various",
                    "rationale": "Safe, guaranteed returns"
                },
                {
                    "name": "Corporate Bonds",
                    "ticker": "Various",
                    "rationale": "Higher yields with moderate risk"
                }
            ]
        },
        "Gold": {
            "current_trend": "Safe haven asset",
            "outlook": "Hedge against uncertainty",
            "key_factors": [
                "Global economic conditions",
                "Currency movements"
            ],
            "risks": [
                "Price volatility",
                "No regular income"
            ],
            "specific_suggestions": [
                {
                    "name": "Gold ETFs",
                    "ticker": "Various",
                    "rationale": "Liquid gold investment"
                },
                {
                    "name": "Sovereign Gold Bonds",
                    "ticker": "Various",
                    "rationale": "Government-backed gold investment"
                }
            ]
        }
    }
    return market_trends.get(asset_type, {})

def get_default_market_data(asset_type: str) -> Dict:
    """Provide default market data when API fails"""
    return {
        "current_trend": f"Analysis for {asset_type} not available",
        "outlook": "Data currently unavailable",
        "key_factors": ["Market data being updated"],
        "risks": ["Data temporarily unavailable"],
        "top_picks": [{"name": "Data unavailable", "symbol": "N/A", "performance": "N/A"}]
    }

async def get_real_time_market_data(asset_type: str) -> Dict:
    """Fetch real-time market data using Perplexity API"""
    url = "https://api.perplexity.ai/chat/completions"
    
    queries = {
        "Stocks": "Analyze current Indian stock market conditions including Nifty50, top performing stocks, and market trends.",
        "Mutual_Funds": "List top performing mutual funds in India with their current returns and trends.",
        "ETFs": "Analyze popular ETFs in India including their performance and market trends."
    }
    
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a financial analyst. Provide a concise market analysis."
            },
            {
                "role": "user",
                "content": queries.get(asset_type, f"Analyze {asset_type} market in India")
            }
        ],
        "temperature": 0.2
    }
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Perplexity API error: {response.status_code}")
                return get_default_market_data(asset_type)
    except Exception as e:
        logger.error(f"Error in get_real_time_market_data: {str(e)}")
        return get_default_market_data(asset_type)

def structure_market_data(raw_data: Dict, asset_type: str) -> Dict:
    """Structure market data using OpenAI"""
    try:
        prompt = f"""Structure the following market analysis for {asset_type} into a JSON format with:
        - current_trend (string)
        - outlook (string)
        - key_factors (list of strings)
        - risks (list of strings)
        - top_picks (list of objects with name, symbol, performance)
        
        Data: {json.dumps(raw_data)}"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial data structuring assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in structure_market_data: {str(e)}")
        return get_default_market_data(asset_type)

async def get_market_analysis(asset_type: str) -> Dict:
    """Get market analysis with fallback mechanism"""
    try:
        if not PERPLEXITY_API_KEY:
            logger.warning("No Perplexity API key found, using fallback data")
            return get_fallback_market_analysis(asset_type)

        logger.info(f"\n{'='*50}")
        logger.info(f"Starting analysis for {asset_type}")
        logger.info(f"Using Perplexity API key: {PERPLEXITY_API_KEY[:10]}...")
        url = "https://api.perplexity.ai/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        queries = {
            "Stocks": """Analyze current Indian stock market conditions. Include:
                1. Current market trend and sentiment
                2. Key market indices performance
                3. Top performing sectors
                4. Recommended large-cap and mid-cap stocks
                Format as JSON with: current_trend, outlook, key_factors, risks, specific_suggestions""",
            "Mutual_Funds": """Analyze current Indian mutual fund market. Include:
                1. Top performing fund categories
                2. Best rated mutual funds
                3. Risk factors and outlook
                4. Specific fund recommendations
                Format as JSON with: current_trend, outlook, key_factors, risks, specific_suggestions""",
            "Bonds": """Analyze current Indian bond market conditions. Include:
                1. Interest rate environment
                2. Government and corporate bond performance
                3. Key risk factors
                4. Specific bond recommendations
                Format as JSON with: current_trend, outlook, key_factors, risks, specific_suggestions""",
            "Gold": """Analyze current gold market conditions. Include:
                1. Price trends and outlook
                2. Key factors affecting gold prices
                3. Investment options in gold
                4. Specific recommendations for gold investments
                Format as JSON with: current_trend, outlook, key_factors, risks, specific_suggestions"""
        }
        
        query = queries.get(asset_type, f"Analyze {asset_type} market in India")
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a financial analyst. Provide analysis in this exact JSON format:
                    {
                        "current_trend": "string",
                        "outlook": "string",
                        "key_factors": ["string"],
                        "risks": ["string"],
                        "specific_suggestions": [
                            {
                                "name": "string",
                                "ticker": "string",
                                "rationale": "string"
                            }
                        ]
                    }"""
                },
                {"role": "user", "content": query}
            ]
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"\n--- Making API request for {asset_type} ---")
            logger.info(f"Request URL: {url}")
            logger.info(f"Request Headers: {headers}")
            logger.info(f"Request Payload: {json.dumps(payload, indent=2)}")
            
            try:
                logger.info(f"\n-> Sending request for {asset_type}...")
                response = await client.post(url, json=payload, headers=headers)
                logger.info(f"<- Received response for {asset_type} (Status: {response.status_code})")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"\n=== Processing response for {asset_type} ===")
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        try:
                            logger.info(f"\nRaw content for {asset_type}:")
                            logger.debug(content)
                            
                            # Strip markdown code blocks if present
                            content = content.strip()
                            if content.startswith('```json'):
                                content = content[7:]  # Remove ```json
                                logger.info(f"Removed ```json prefix for {asset_type}")
                            if content.endswith('```'):
                                content = content[:-3]  # Remove ```
                                logger.info(f"Removed ``` suffix for {asset_type}")
                            content = content.strip()  # Remove any extra whitespace
                            
                            # Clean up citations in the content
                            import re
                            content = re.sub(r'\[\d+\]', '', content)  # Remove citation references like [3], [4]
                            logger.info(f"\nCleaned content for {asset_type}:")
                            logger.debug(content)  # Use debug level for large content
                            
                            logger.info(f"\nParsing JSON for {asset_type}...")
                            market_data = json.loads(content)
                            logger.info(f"Successfully parsed JSON for {asset_type}")
                            logger.info(f"Market data: {json.dumps(market_data, indent=2)}")
                            
                            required_keys = ['current_trend', 'outlook', 'key_factors', 'risks', 'specific_suggestions']
                            if all(key in market_data for key in required_keys):
                                logger.info(f"All required keys present for {asset_type}")
                                if isinstance(market_data['specific_suggestions'], list):
                                    for suggestion in market_data['specific_suggestions']:
                                        if not all(key in suggestion for key in ['name', 'ticker', 'rationale']):
                                            logger.error(f"Invalid suggestion format in {asset_type}: {suggestion}")
                                            raise ValueError(f"Invalid suggestion format in {asset_type}")
                                    logger.info(f"Successfully validated {asset_type} data")
                                    return market_data
                            else:
                                logger.error(f"Missing required keys in {asset_type} data")
                            
                            logger.error(f"Invalid data structure from API for {asset_type}")
                        except json.JSONDecodeError as e:
                            logger.error(f"\nJSON parsing error for {asset_type}: {str(e)}")
                            logger.error(f"Problematic content for {asset_type}:")
                            logger.error(content)
                        except Exception as e:
                            logger.error(f"\nError processing market data for {asset_type}: {str(e)}")
                            logger.error(f"Problematic content for {asset_type}:")
                            logger.error(content)
                else:
                    logger.error(f"\nAPI error for {asset_type}:")
                    logger.error(f"Status code: {response.status_code}")
                    logger.error(f"Response text: {response.text}")
                    logger.error(f"Response headers: {response.headers}")
            except httpx.TimeoutException:
                logger.error(f"\nRequest timeout for {asset_type} after 30 seconds")
            except Exception as e:
                logger.error(f"\nRequest error for {asset_type}: {str(e)}")
                import traceback
                logger.error(f"Full error traceback for {asset_type}:")
                logger.error(traceback.format_exc())
            
            logger.info(f"\nFalling back to default market analysis for {asset_type}")
            return get_fallback_market_analysis(asset_type)
    except Exception as e:
        logger.error(f"\nUnexpected error in get_market_analysis for {asset_type}: {str(e)}")
        import traceback
        logger.error(f"Full error traceback for {asset_type}:")
        logger.error(traceback.format_exc())
        return get_fallback_market_analysis(asset_type)

def get_investment_recommendations(market_data: Dict, risk_profile: str) -> List[Dict]:
    """Generate specific investment recommendations based on market data and risk profile."""
    try:
        recommendations = []
        asset_types = ["Stocks", "Mutual_Funds", "Bonds", "Gold"]
        
        for asset_type in asset_types:
            market_info = market_data.get(asset_type, {})
            specific_suggestions = market_info.get("specific_suggestions", [])
            
            if not specific_suggestions:
                if asset_type == "Stocks":
                    if risk_profile == "Conservative":
                        specific_suggestions = [
                            {"name": "Blue-chip Dividend Stocks", "ticker": "Various", "rationale": "Stable companies with consistent dividend payments"},
                            {"name": "Consumer Staples ETF", "ticker": "Various", "rationale": "Defensive sector with stable earnings"}
                        ]
                    else:
                        specific_suggestions = [
                            {"name": "Growth Stocks", "ticker": "Various", "rationale": "High-growth potential companies"},
                            {"name": "Technology Sector ETF", "ticker": "Various", "rationale": "Exposure to tech innovation"}
                        ]
                elif asset_type == "Mutual_Funds":
                    specific_suggestions = [
                        {"name": "Index Funds", "ticker": "Various", "rationale": "Low-cost market exposure"},
                        {"name": "Balanced Funds", "ticker": "Various", "rationale": "Mix of stocks and bonds"}
                    ]
                elif asset_type == "Bonds":
                    specific_suggestions = [
                        {"name": "Government Bonds", "ticker": "Various", "rationale": "Safe, guaranteed returns"},
                        {"name": "Corporate Bonds", "ticker": "Various", "rationale": "Higher yields with moderate risk"}
                    ]
                elif asset_type == "Gold":
                    specific_suggestions = [
                        {"name": "Gold ETFs", "ticker": "Various", "rationale": "Liquid gold investment"},
                        {"name": "Sovereign Gold Bonds", "ticker": "Various", "rationale": "Government-backed gold investment"}
                    ]

            allocation = get_allocation_percentage(asset_type, risk_profile)
            
            recommendations.append({
                "investment_type": asset_type,
                "allocation_percentage": allocation,
                "details": f"Recommended allocation for {asset_type.replace('_', ' ').lower()}",
                "specific_suggestions": specific_suggestions,
                "entry_strategy": get_entry_strategy(asset_type, risk_profile),
                "exit_strategy": get_exit_strategy(asset_type, risk_profile),
                "risk_mitigation": get_risk_mitigation(asset_type)
            })

        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return get_default_recommendations(risk_profile)

def get_allocation_percentage(asset_type: str, risk_profile: str) -> float:
    """Determine allocation percentage based on asset type and risk profile."""
    allocations = {
        "Conservative": {
            "Stocks": 20.0,
            "Mutual_Funds": 25.0,
            "Bonds": 40.0,
            "Gold": 15.0
        },
        "Moderate": {
            "Stocks": 35.0,
            "Mutual_Funds": 30.0,
            "Bonds": 25.0,
            "Gold": 10.0
        },
        "Aggressive": {
            "Stocks": 50.0,
            "Mutual_Funds": 30.0,
            "Bonds": 15.0,
            "Gold": 5.0
        }
    }
    return allocations.get(risk_profile, allocations["Moderate"]).get(asset_type, 25.0)

def get_entry_strategy(asset_type: str, risk_profile: str) -> str:
    """Generate entry strategy based on asset type and risk profile."""
    strategies = {
        "Stocks": "Use dollar-cost averaging to enter positions gradually",
        "Mutual_Funds": "Systematic investment plan (SIP) for regular investments",
        "Bonds": "Ladder strategy with staggered maturity dates",
        "Gold": "Regular small purchases to average out price fluctuations"
    }
    return strategies.get(asset_type, "Systematic and gradual entry")

def get_exit_strategy(asset_type: str, risk_profile: str) -> str:
    """Generate exit strategy based on asset type and risk profile."""
    strategies = {
        "Stocks": "Set target prices and trailing stop losses",
        "Mutual_Funds": "Review and rebalance quarterly",
        "Bonds": "Hold till maturity unless significant market changes",
        "Gold": "Maintain as long-term hedge, sell partial positions at significant highs"
    }
    return strategies.get(asset_type, "Regular review and rebalancing")

def get_risk_mitigation(asset_type: str) -> str:
    """Generate risk mitigation strategy based on asset type."""
    strategies = {
        "Stocks": "Diversification across sectors and market caps",
        "Mutual_Funds": "Mix of different fund types and investment styles",
        "Bonds": "Diversify across different issuers and maturities",
        "Gold": "Maintain as portfolio hedge, limit exposure to 5-15%"
    }
    return strategies.get(asset_type, "Diversification and regular monitoring")

def get_default_recommendations(risk_profile: str) -> List[Dict]:
    """Provide default recommendations when API calls fail."""
    return [
        {
            "investment_type": "Stocks",
            "allocation_percentage": 30.0,
            "details": "Default stock allocation",
            "specific_suggestions": [
                {"name": "Large Cap Index Fund", "ticker": "Various", "rationale": "Market stability"}
            ],
            "entry_strategy": "Dollar-cost averaging",
            "exit_strategy": "Regular rebalancing",
            "risk_mitigation": "Diversification across sectors"
        }
    ]

@app.post("/analyze-portfolio", response_model=PortfolioRecommendation)
async def analyze_portfolio(user_data: UserProfile) -> PortfolioRecommendation:
    """Analyze user profile and generate portfolio recommendations."""
    try:
        logger.info("\nStarting portfolio analysis...")
        logger.info(f"User profile: {user_data.dict()}")
        market_analysis = {}
        for asset_type in ["Stocks", "Mutual_Funds", "Bonds", "Gold"]:
            market_analysis[asset_type] = await get_market_analysis(asset_type)
        
        recommendations = get_investment_recommendations(market_analysis, user_data.risk_tolerance)
        
        explanation = (
            f"Based on your {user_data.risk_tolerance} risk profile and current market conditions, "
            f"we have prepared a personalized investment portfolio recommendation. "
            f"The allocation considers your investment horizon of {user_data.investment_horizon} "
            f"and comfort with fluctuations level of {user_data.comfort_with_fluctuations}."
        )
        
        review_schedule = "Quarterly" if user_data.investment_horizon == "Short" else "Semi-annually"
        
        portfolio = PortfolioRecommendation(
            explanation=explanation,
            recommendations=recommendations,
            market_analysis=market_analysis,
            review_schedule=review_schedule,
            disclaimer=(
                "This recommendation is based on current market conditions and your profile. "
                "Past performance is not indicative of future results. "
                "Please consult with a financial advisor before making investment decisions."
            )
        )
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error in analyze_portfolio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating portfolio recommendation: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
