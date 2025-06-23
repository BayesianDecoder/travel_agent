



from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
import os
import streamlit as st


# Groq LLM
groq_llm = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY", "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")
)

# Simple search function 
def search_web_tool(query: str) -> str:
    search = DuckDuckGoSearchResults(num_results=3)
    try:
        raw_results = search.run(query)  # This is a plain string or list of links
    except Exception as e:
        return "No search results found."

    if isinstance(raw_results, str):
        return raw_results  # just return raw string
    elif isinstance(raw_results, list):
        return "\n".join([f"{i+1}. {r.get('title', '')}: {r.get('link', '')}" for i, r in enumerate(raw_results)])
    else:
        return "Unknown format for search results."

#  Prompt templates
guide_prompt = PromptTemplate.from_template("""
You are a travel city guide for {destination_city}.
Create a markdown itinerary for a visitor arriving on {date_from} and leaving on {date_to}.

They are interested in: {interests}.
Suggest activities, food, and local events. All suggestions must be budget-friendly under {budget} rupees.

Use markdown with bold headings and bullet points for clarity.
""")

location_prompt = PromptTemplate.from_template("""
You are a travel expert helping someone travel from {from_city} to {destination_city}
between {date_from} and {date_to}. The user's total budget is {budget} rupees.

Provide the following:
- Local accommodation types & estimated cost
- Cost of living
- Visa requirements(if required)
- Transportation options
- Weather forecast
- Any major local events during the visit

Format the output as markdown with headers and bullet points.
""")

planner_prompt = PromptTemplate.from_template("""
You are a professional travel planner.

Here is information gathered from two experts:

== Travel Logistics ==
{location_info}

== City Activities and Food ==
{guide_info}

== Real-Time Travel Tips ==
{live_info}

Using this and your own knowledge, create a well-structured travel itinerary including:
- Best accommodations **strictly** under {budget}. If not in {budget} , give recommendations.
- Visa and weather advice
- Transportation
- Cost of living
- Any major local events

Include:
- 4-paragraph city intro
- a **Detailed** Day-wise plan with time blocks and make sure the places are not repetitive to vist
- Tips on transport, food, events
- Budget breakdown under {budget} Rupees

Use markdown formatting.
""")

# ✅ Chains
guide_chain = guide_prompt | groq_llm
location_chain = location_prompt | groq_llm
planner_chain = planner_prompt | groq_llm

# ✅ LangGraph nodes
def location_node(state: dict) -> dict:
    query = f"travel tips {state['destination_city']} {state['date_from']} to {state['date_to']}"
    live_info = search_web_tool(query)

    response = location_chain.invoke({
        "from_city": state["from_city"],
        "destination_city": state["destination_city"],
        "date_from": state["date_from"],
        "date_to": state["date_to"],
        "budget": state["budget"],
        "live_info": live_info
    })

    return {**state, "location_info": response.content}

def guide_node(state: dict) -> dict:
    response = guide_chain.invoke(state)
    return {**state, "guide_info": response.content}

def planner_node(state: dict) -> dict:
    if "guide_info" not in state or "location_info" not in state:
        raise ValueError("Missing guide_info or location_info in state")

    live_info = search_web_tool(f"latest travel tips and events in {state['destination_city']} 2025")

    full_input = {
        "guide_info": state["guide_info"],
        "location_info": state["location_info"],
        "live_info": live_info,
        "budget": state.get("budget", "unknown")
    }

    result = planner_chain.invoke(full_input)
    return {**state, "final_itinerary": result.content}



# --- BUILD LANGGRAPH
graph = StateGraph(dict)
graph.add_node("location", RunnableLambda(location_node))
graph.add_node("guide", RunnableLambda(guide_node))
graph.add_node("planner", RunnableLambda(planner_node))

graph.set_entry_point("location")
graph.add_edge("location", "guide")
graph.add_edge("guide", "planner")
graph.add_edge("planner", END)

travel_graph = graph.compile()

# --- STREAMLIT UI
st.title("🌍 AI-Powered Travel Planner")
st.markdown("Plan your trip with a city guide, logistics, and full itinerary in markdown!")

with st.form("trip_form"):
    from_city = st.text_input("🏡 From City", "Delhi")
    destination_city = st.text_input("✈️ Destination City", "Singapore")
    date_from = st.date_input("📅 Departure Date")
    date_to = st.date_input("📅 Return Date")
    interests = st.text_area("🎯 Your Interests", "food, adventure, markets")
    budget = st.number_input("💰 Budget (₹)", min_value=5000, step=500, value=30000)
    submitted = st.form_submit_button("🚀 Generate Travel Plan")

if submitted:
    with st.spinner("🧠 Generating AI-powered travel plan..."):
        result = travel_graph.invoke({
            "from_city": from_city,
            "destination_city": destination_city,
            "date_from": str(date_from),
            "date_to": str(date_to),
            "interests": interests,
            "budget": budget
        })

        plan = result["final_itinerary"]

        st.success("✅ Here's your travel itinerary!")
        st.markdown(plan)

        st.download_button(
            label="📥 Download Itinerary (.md)",
            data=plan,
            file_name=f"TravelPlan_{destination_city}.md",
            mime="text/markdown"
        )