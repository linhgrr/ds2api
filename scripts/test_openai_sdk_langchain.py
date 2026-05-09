import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1️⃣ Define structured schema
class PersonInfo(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age")
    job: str = Field(description="The person's job title")

# 2️⃣ Setup OpenAI-compatible LLM
llm = ChatOpenAI(
    model="gpt-5.5",
    base_url="http://localhost:5002/v1",  # 👈 OpenAI-compatible endpoint
    api_key="proxypal-local",
    temperature=0
)

# 3️⃣ Enable structured output
structured_llm = llm.with_structured_output(PersonInfo)

# 4️⃣ Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured information from the user input."),
    ("human", "{input}")
])

chain = prompt | structured_llm

# 5️⃣ Test
result = chain.invoke({
    "input": "John Smith is a 32 year old software engineer."
})

print(result)
print(type(result))