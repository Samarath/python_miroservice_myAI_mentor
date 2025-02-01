import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime


#setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            'roadmap_api.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)


app = FastAPI()
logger = logging.getLogger("roadmap_api")

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
model = GoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY)



def get_ai(topic):
    prompt = PromptTemplate.from_template(
        "Create a structured learning roadmap for {concept}. Break it into key milestones, estimate the time required for each stage, and outline essential topics. Suggest resources (books, courses, websites) and practical exercises to reinforce learning. Keep it clear, actionable, and goal-oriented.")
    chain = {"concept": RunnablePassthrough()} | prompt | model
    response = chain.invoke(topic)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )

@app.get("/")
async def home():
    logger.info("Root endpoint accessed")
    return {
            "status": 200,
            "timestamp": datetime.utcnow().isoformat(),
            "response": {"topic":"", "data": "API is working",}
        }

@app.get("/roadmap/{topic}")
async def get_generated_response(topic: str):
    # input validation
    if not topic or len(topic) > 100:
        logger.warning(f"Invalid topic length: {topic}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Topic must be between 1-100 characters"
        )
    try:
        logger.info(f"Generating roadmap for {topic}")
        response = get_ai(topic)
        logger.info(f"Successfully generated roadmap for {topic}")
        return {
            "status": 200,
            "timestamp": datetime.utcnow().isoformat(),
            "response": {"topic":topic, "data": response}
        }
    except Exception as e:
        logger.error(f"Error generating roadmap for {topic}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
             detail={
                "status": 500,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Failed to generate roadmap: {str(e)}"
            }
        )
