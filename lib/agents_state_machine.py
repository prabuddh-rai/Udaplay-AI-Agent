from typing import TypedDict, List, Optional, Union, TypeVar
import json

import chromadb 
from chromadb.utils import embedding_functions
from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory, LongTermMemory, MemoryFragment

# Define the state schema
class AgentState(TypedDict):
    user_query: str  # The current user query being processed
    instructions: str  # System instructions for the agent
    messages: List[dict]  # List of conversation messages
    retrieved_docs: List[dict] # List of games stored as dictionaries
    evaluation: bool # Boolean value depending on evaluation results
    evaluation_result: str # Final evaluation result
    web_search_results: str # Results from Web Seacrh
    final_answer: str # Final AI Message answer
    recalled_memory: List[MemoryFragment] # Recalled Memory Fragment saved in Long Term Memory(LTM)
    total_tokens: int  # Track the cumulative total
    
class Agent:
    def __init__(self, 
                 model_name: str,
                 instructions: str, 
                 temperature: float = 0.7,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 tools: List[Tool] = None):
        """
        Initialize an Agent
        
        Args:
            model_name: Name/identifier of the LLM model to use
            instructions: System instructions for the agent
            tools: Optional list of tools available to the agent
            temperature: Temperature parameter for LLM (default: 0.7)
        """
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        
        # Initialize Short Term Memory, Long Term Memory and State Machine
        self.memory = ShortTermMemory()
        self.chroma_client = chromadb.PersistentClient(path="chromadb")
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key = self.api_key,
            api_base = self.base_url,
            model_name="text-embedding-3-small"
        )
        self.ltm_memory = LongTermMemory(
            client=self.chroma_client,
            name="udaplay_longterm",
            embedding_fn = self.embedding_fn
        )
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        """Step logic: Prepare messages for LLM consumption"""
        messages = state.get("messages", [])
        
        # If no messages exist, start with system message
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]
            
        # Add the new user message
        messages.append(UserMessage(content=state["user_query"]))

        return {
            "messages": messages,
            "session_id": state["session_id"],
        }

    def _recall_memory_step(self, state: AgentState) -> AgentState:
        """Step logic: Retrieve memory from vector DB"""
        hits = self.ltm_memory.search(
            state["user_query"], 
            limit=1
        )
        
        fragments = hits.fragments
        distances = hits.metadata.get("distances", [])

        #Only trusting a strong match 
        recalled = fragments[0].content if fragments and distances and distances[0] < 0.35 else None
        
        return {
        **state,
        "recalled_memory": recalled,
        }
    
    def _retrieve_game_step(self, state: AgentState) -> AgentState:
        """Step logic: Retrieve game information from vector DB"""
        tool = self.get_tool("retrieve_game")

        result = str(
            tool(query=state["user_query"])
            )

        return {
        **state,
        "retrieved_docs": result,
        }

    def _evaluate_retrieval_step(self, state: AgentState) -> AgentState:
         """Step logic: Evaluate retrieved documents"""
         tool = self.get_tool("evaluate_retrieval")

         result = str(
            tool(
                question=state["user_query"],
                retrieved_docs=state["retrieved_docs"]
            )
         )
         parsed_result = json.loads(result)

         return {
            **state,
            "evaluation": parsed_result.get("useful", False),
            "evaluation_result": result,
         }

    def _web_search_step(self, state: AgentState) -> AgentState:
        """Step logic: Search web for additional game information"""
        tool = self.get_tool("game_web_search")
        result = str(
            tool(question=state["user_query"])
        )

        return{
            **state,
            "web_search_results": result,
        }

    def _persist_memory_step(self, state: AgentState) -> AgentState:
        """Step logic: Register new memory to vector DB"""
        if state.get("web_search_results"):
            fragment = MemoryFragment(
                content=state["web_search_results"],
                owner="default",
                namespace="web_search"
            )

        self.ltm_memory.register(
            memory_fragment=fragment,
            metadata={"question": state["user_query"]}
        )
        
        return {
            **state,
        }

    def _final_answer_step(self, state: AgentState) -> AgentState:
        """Step logic: Generate final answer"""
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            base_url=self.base_url
        )

        final_prompt = f"""
        {state["instructions"]}

        User Question:
        {state["user_query"]}

        Internal Retrieved Memory:
        {state.get("recalled_memory", "")}

        Internal Retrieved Documents:
        {state.get("retrieved_docs", "")}

        Retrieval Evaluation:
        {state.get("evaluation_result", "")}

        Web Search Results:
        {state.get("web_search_results", "")}

        Using the available information above, provide a clear final answer.
        If the internal database was sufficient, cite the internal database.
        If web search was used, cite web search results.
        """
        response = llm.invoke(final_prompt)

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        # Create AI message with content and tool calls
        ai_message = AIMessage(
            content=response.content, 
        )
        return {
            **state,
            "messages": state["messages"] + [ai_message],
            "final_answer": response.content,
            "total_tokens": current_total,
        }
        

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """Create the internal state machine for the agent"""
        machine = StateMachine[AgentState](AgentState)

        
        # Create steps
        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        recall_memory = Step[AgentState]("recall_memory", self._recall_memory_step)
        game_retrieval = Step[AgentState]("game_retrieval", self._retrieve_game_step)
        answer_evaluator = Step[AgentState]("answer_evaluator", self._evaluate_retrieval_step)
        web_search = Step[AgentState]("web_search", self._web_search_step)
        persist_memory = Step[AgentState]("persist_memory", self._persist_memory_step)
        final_answer = Step[AgentState]("final_answer", self._final_answer_step)
        termination = Termination[AgentState]()
        
        machine.add_steps([entry, message_prep, recall_memory, game_retrieval, answer_evaluator, web_search, persist_memory, final_answer, termination])
        
        # Add transitions
        machine.connect(entry, message_prep)
        machine.connect(message_prep, recall_memory)
         
        # Transition based on recall_memory step
        def check_recalled_memory(state: AgentState) -> Union[Step[AgentState], str]:
            """Transition logic: Check if there is recalled memory"""
            if state.get("recalled_memory"):
                return final_answer
            return game_retrieval
        
        machine.connect(recall_memory, [final_answer, game_retrieval], check_recalled_memory) # Transition Step1
        machine.connect(game_retrieval, answer_evaluator)
        
        # Transition based on answer_evaluator step
        def check_evaluation(state: AgentState) -> Union[Step[AgentState], str]:
            """Transition logic: Check if evaluation is True"""
            if state.get("evaluation"):
                return final_answer
            return web_search
        
        machine.connect(answer_evaluator, [final_answer, web_search], check_evaluation) # Transition Step2
        machine.connect(web_search, persist_memory) # Registering to LTM
        machine.connect(persist_memory, final_answer) # Generate a final AI message response
        machine.connect(final_answer, termination) 
        
        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a query
        
        Args:
            query: The user's query to process
            session_id: Optional session identifier (uses "default" if None)
            
        Returns:
            The final run object after processing
        """
        session_id = session_id or "default"

        # Create session if it doesn't exist
        self.memory.create_session(session_id)

        # Get previous messages from last run if available
        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "evaluation": False,
            "evaluation_result": "",
            "session_id": session_id,
            "retrieved_docs": "",
            "web_search_results": "",
            "final_answer": "",
            "recalled_memory": None
        }

        run_object = self.workflow.run(initial_state)
        
        # Store the complete run object in memory
        self.memory.add(run_object, session_id)
        
        return run_object

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool object by its name

        Args:
        tool_name: Name of the tool to retrieve

        Returns:
        Tool object if found, otherwise None
        """

        return next(
           (tool for tool in self.tools if tool.name == tool_name),
           None
        )

    def reset_session(self, session_id: Optional[str] = None):
        """Reset memory for a specific session
        
        Args:
            session_id: Optional session to reset (uses "default" if None)
        """
        self.memory.reset(session_id)
