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
import re
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def make_api_request(client: httpx.AsyncClient, url: str, headers: dict, payload: dict, asset_type: str):
    """Make API request with retry mechanism"""
    logger.info(f"\n-> Sending request for {asset_type}...")
    response = await client.post(url, json=payload, headers=headers, timeout=60.0)
    logger.info(f"<- Received response for {asset_type} (Status: {response.status_code})")
    
    if response.status_code != 200:
        logger.error(f"API error for {asset_type}: Status {response.status_code}")
        logger.error(f"Response: {response.text}")
        raise httpx.HTTPError(f"API request failed with status {response.status_code}")
    
    return response.json()

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
                1. Current interest rate environment and trends
                2. Government bond yields and performance
                3. Corporate bond market overview
                4. Specific recommendations for both government and corporate bonds
                Format response as a JSON object with these exact keys:
                {
                    "current_trend": "description of current bond market trends",
                    "outlook": "future outlook for bonds",
                    "key_factors": ["list of key factors affecting bonds"],
                    "risks": ["list of main risks"],
                    "specific_suggestions": [
                        {
                            "name": "name of bond or bond fund",
                            "ticker": "ticker symbol if applicable",
                            "rationale": "why this is recommended"
                        }
                    ]
                }""",
            "Gold": """Analyze current gold market conditions. Include:
                1. Price trends and outlook
                2. Key factors affecting gold prices
                3. Investment options in gold
                4. Specific recommendations for gold investments
                Format as JSON with: current_trend, outlook, key_factors, risks, specific_suggestions"""
        }
        
        query = queries.get(asset_type, f"Analyze {asset_type} market in India")
        
        payload = {
            "model": "mistral-7b-instruct",  # Using Mistral model for better structured output
            "messages": [
                {
                    "role": "system",
                    "content": """You are a financial analyst. Always respond with valid JSON in this exact format:
                    {
                        "current_trend": "string describing current market trend",
                        "outlook": "string describing future outlook",
                        "key_factors": ["list of key market factors"],
                        "risks": ["list of key risks"],
                        "specific_suggestions": [
                            {
                                "name": "investment name",
                                "ticker": "ticker symbol or identifier",
                                "rationale": "reason for recommendation"
                            }
                        ]
                    }
                    Do not include any markdown formatting or extra text."""
                },
                {"role": "user", "content": query}
            ]
        }
        
        async with httpx.AsyncClient() as client:
            logger.info(f"\n--- Making API request for {asset_type} ---")
            logger.info(f"Request URL: {url}")
            logger.info(f"Request Headers: {headers}")
            logger.info(f"Request Payload: {json.dumps(payload, indent=2)}")
            
            try:
                result = await make_api_request(client, url, headers, payload, asset_type)
                
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    try:
                        logger.info(f"\nRaw content for {asset_type}:")
                        logger.debug(content)
                        
                        # Clean up the content
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content[7:]
                            logger.info(f"Removed ```json prefix for {asset_type}")
                        if content.endswith('```'):
                            content = content[:-3]
                            logger.info(f"Removed ``` suffix for {asset_type}")
                        content = content.strip()
                        
                        # Remove citations and clean up
                        content = re.sub(r'\[\d+\]', '', content)
                        logger.info(f"\nCleaned content for {asset_type}:")
                        logger.debug(content)
                        
                        # Parse JSON with error handling
                        try:
                            market_data = json.loads(content)
                            logger.info(f"Successfully parsed JSON for {asset_type}")
                            
                            # Validate required fields
                            required_keys = ['current_trend', 'outlook', 'key_factors', 'risks', 'specific_suggestions']
                            if all(key in market_data for key in required_keys):
                                if isinstance(market_data['specific_suggestions'], list):
                                    for suggestion in market_data['specific_suggestions']:
                                        if all(key in suggestion for key in ['name', 'ticker', 'rationale']):
                                            continue
                                        logger.error(f"Invalid suggestion format in {asset_type}: {suggestion}")
                                        raise ValueError(f"Invalid suggestion format in {asset_type}")
                                    logger.info(f"Successfully validated {asset_type} data")
                                    return market_data
                            
                            logger.error(f"Missing required keys in {asset_type} data")
                            raise ValueError(f"Invalid data structure for {asset_type}")
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"\nJSON parsing error for {asset_type}: {str(e)}")
                            logger.error(f"Problematic content: {content}")
                            raise
                            
                    except Exception as e:
                        logger.error(f"\nError processing {asset_type} data: {str(e)}")
                        raise
                
                logger.error(f"Invalid API response format for {asset_type}")
                raise ValueError("Invalid API response format")
                
            except Exception as e:
                logger.error(f"\nError in market analysis for {asset_type}: {str(e)}")
                logger.info(f"Falling back to default market analysis for {asset_type}")
                return get_fallback_market_analysis(asset_type)
                
    except Exception as e:
        logger.error(f"\nUnexpected error in get_market_analysis for {asset_type}: {str(e)}")
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
                        {"name": "Government Securities", "ticker": "Various", "rationale": "Safe, guaranteed returns"},
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
        
        # Generate detailed explanation using market analysis
        explanation = []
        explanation.append(
            f"Based on your {user_data.risk_tolerance} risk profile and current market conditions, "
            f"we have prepared a personalized investment portfolio recommendation. "
            f"The allocation considers your investment horizon of {user_data.investment_horizon} "
            f"and comfort with fluctuations level of {user_data.comfort_with_fluctuations}."
        )
        
        # Add market insights
        explanation.append("\nMarket Analysis:")
        for asset_type, analysis in market_analysis.items():
            explanation.append(f"\n{asset_type}:")
            explanation.append(f"• Current Trend: {analysis['current_trend']}")
            explanation.append(f"• Outlook: {analysis['outlook']}")
        
        # Add allocation strategy
        explanation.append("\nAllocation Strategy:")
        for rec in recommendations:
            explanation.append(
                f"\n• {rec['investment_type']} ({rec['allocation_percentage']}%): {rec['details']}"
            )
        
        # Add risk considerations based on user profile
        explanation.append(f"\nRisk Considerations:")
        for asset_type, analysis in market_analysis.items():
            if analysis['risks']:
                explanation.append(f"\n• {asset_type} Risks: {', '.join(analysis['risks'])}")
        
        review_schedule = "Quarterly" if user_data.investment_horizon == "Short" else "Semi-annually"
        
        portfolio = PortfolioRecommendation(
            explanation="\n".join(explanation),
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
