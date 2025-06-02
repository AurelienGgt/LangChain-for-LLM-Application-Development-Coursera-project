import os
import time
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate # Explicit import for clarity if needed later
from langchain import hub
from langchain_core.cache import InMemoryCache
from langchain.globals import set_llm_cache

# --- Module Docstring ---
"""
This script implements a basic Langchain agent with a custom tool for string length calculation.
It demonstrates:
1.  Setting up a Langchain agent using ChatOpenAI and a ReAct prompt.
2.  Defining and using custom tools.
3.  Implementing in-memory caching for LLM responses to optimize repeated queries.
4.  A basic testing function to verify agent functionality and caching.

To run this agent, ensure the OPENAI_API_KEY environment variable is set.
"""

# --- Caching Setup ---
# Set up in-memory caching for LLM responses globally for the Langchain library.
# This means that any LLM call within Langchain that is identical to a previous one
# (same model, parameters, and prompt) will return the cached response.
set_llm_cache(InMemoryCache())
print("In-memory LLM caching enabled.")

# --- Tool Definition ---
def string_length_tool_func(input_str: str) -> int:
    """
    Calculates and returns the length of the input string.
    Raises TypeError if the input is not a string.
    """
    if not isinstance(input_str, str):
        raise TypeError("Input must be a string")
    print(f"Tool 'string_length_tool_func' called with: '{input_str}'") # Added for clarity during runs
    return len(input_str)

# The Tool object wraps the function, providing a name and description for the agent to understand its use.
string_length_tool = Tool(
    name="StringLengthTool",
    func=string_length_tool_func,
    description="Useful for when you need to find the length of a string. Input must be a single string.",
)

tools = [string_length_tool]

# --- Agent Setup ---
# Attempt to retrieve the OpenAI API key from environment variables.
# This is a secure way to handle API keys without hardcoding them.
api_key = os.environ.get("OPENAI_API_KEY")
llm = None # Initialize llm to None

if api_key:
    # Initialize the ChatOpenAI model. Temperature 0 is used for more deterministic outputs.
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    print("ChatOpenAI model initialized.")
else:
    # Inform the user if the API key is missing, as the agent relies on it.
    print("WARNING: OPENAI_API_KEY environment variable not set. Agent will not be fully functional.")

agent_executor = None # Initialize agent_executor to None
if llm:
    # Pull a standard ReAct prompt template from Langchain Hub.
    # This prompt guides the LLM on how to use tools and think step-by-step.
    prompt = hub.pull("hwchase17/react")

    # Create the agent using the LLM, tools, and the prompt.
    agent = create_react_agent(llm, tools, prompt)

    # The AgentExecutor is responsible for running the agent and handling the interaction loop.
    # verbose=False by default for cleaner output during tests, can be overridden.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    print("Langchain ReAct agent executor initialized.")
else:
    print("Agent executor not initialized due to missing LLM (API key likely not set).")


# --- Basic Testing/Evaluation ---
def run_tests():
    """
    Runs a series of basic tests to verify the agent's functionality and caching.
    Prints test results to the console.
    """
    print("\n--- Running Basic Tests ---")
    if not agent_executor:
        print("Agent executor not initialized. Skipping tests.")
        print("--- Basic Tests Finished (Skipped) ---")
        return

    # Test 1: Basic functionality - StringLengthTool
    print("\nTest 1: StringLengthTool functionality...")
    test_string = "hello"
    expected_length = 5
    query1 = f"What is the length of the word '{test_string}'?"
    try:
        response = agent_executor.invoke({"input": query1})
        output = response.get("output", "").lower()
        if str(expected_length) in output and "length" in output and test_string in output:
            print(f"Test 1 PASSED: Agent output for '{query1}' contains '{expected_length}'. Output: {output}")
        else:
            print(f"Test 1 FAILED: Agent output for '{query1}' was '{output}', expected to contain '{expected_length}'.")
    except Exception as e:
        print(f"Test 1 FAILED with error: {e}")

    # Test 2: Tool with different casing and natural language query
    print("\nTest 2: Tool with different casing and natural language...")
    test_string_upper = "WORLD"
    expected_length_upper = 5
    query2 = f"how long is the text '{test_string_upper}' please?"
    try:
        response = agent_executor.invoke({"input": query2})
        output = response.get("output", "").lower()
        if str(expected_length_upper) in output and "length" in output and test_string_upper.lower() in output:
            print(f"Test 2 PASSED: Agent output for '{query2}' contains '{expected_length_upper}'. Output: {output}")
        else:
            print(f"Test 2 FAILED: Agent output for '{query2}' was '{output}', expected to contain '{expected_length_upper}'.")
    except Exception as e:
        print(f"Test 2 FAILED with error: {e}")

    # Test 3: Caching behavior
    print("\nTest 3: Caching behavior verification...")
    cached_query = "Tell me the length of the string 'cachetest'."
    expected_length_cache = 9

    original_verbose = agent_executor.verbose
    agent_executor.verbose = True  # Enable verbose to observe LLM call details for caching

    print("\nFirst call (populating cache):")
    start_time1 = time.time()
    try:
        response1 = agent_executor.invoke({"input": cached_query})
        output1 = response1.get("output", "").lower()
        duration1 = time.time() - start_time1
        print(f"Call 1 for '{cached_query}'. Duration: {duration1:.4f}s. Output: {output1}")
        if not (str(expected_length_cache) in output1 and "length" in output1 and "cachetest" in output1):
             print(f"Warning: Output for first call of Test 3 might be unexpected: {output1}")
    except Exception as e:
        print(f"Test 3 (Call 1 for '{cached_query}') FAILED with error: {e}")
        agent_executor.verbose = original_verbose
        print("--- Basic Tests Finished (Due to Error) ---")
        return

    print("\nSecond call (should hit cache, check verbose output for no LLM calls):")
    start_time2 = time.time()
    try:
        response2 = agent_executor.invoke({"input": cached_query})
        output2 = response2.get("output", "").lower()
        duration2 = time.time() - start_time2
        print(f"Call 2 for '{cached_query}'. Duration: {duration2:.4f}s. Output: {output2}")

        if output1 == output2:
            print("Test 3 PASSED (Conceptual): Second call output matches first.")
            if duration2 < duration1 * 0.9: # Check if significantly faster (allow some margin)
                print(f"  - Second call was noticeably faster ({duration2:.4f}s vs {duration1:.4f}s).")
            else:
                print(f"  - Second call duration ({duration2:.4f}s) vs first ({duration1:.4f}s). Speed difference may not be significant for fast queries, but LLM calls should be absent in verbose logs.")
            print("  - Confirm absence of new LLM calls in the verbose output above for the second call to verify caching.")
        else:
            print(f"Test 3 FAILED (Conceptual): Outputs differ. Output1: {output1}, Output2: {output2}")
    except Exception as e:
        print(f"Test 3 (Call 2 for '{cached_query}') FAILED with error: {e}")
    finally:
        agent_executor.verbose = original_verbose # Reset verbose setting

    print("\n--- Basic Tests Finished ---")

if __name__ == "__main__":
    # The script will primarily run the tests when executed directly.
    # To interact with the agent manually, you might want to comment out run_tests()
    # and uncomment a direct agent_executor.invoke call.
    # Example:
    # if agent_executor:
    #    user_query = "What is the length of 'example'?"
    #    print(f"\nRunning manual query: {user_query}")
    #    response = agent_executor.invoke({"input": user_query})
    #    print("Agent Response:", response)
    # else:
    #    print("Cannot run manual query, agent_executor not initialized.")

    run_tests()
