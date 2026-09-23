import os
import random

from langchain_openai import ChatOpenAI
from langchain.tools import tool




class HRI():
    # Configure the OpenAI client to use Ollama Cloud's OpenAI-compatible endpoint.
    

    def __init__(self):
        self.vlm = ChatOpenAI(
                model="gemma4:cloud",
                api_key=os.environ["OLLAMA_API_KEY"],
                base_url="https://ollama.com/v1",
        )

        self.system_prompt = open("src/roboastra/prompt/hri-system-prompt.md").read()


    @staticmethod
    @tool
    def call_specialist(specialist_name: str = None):
        """Call a specialist robot-action model by its name.

        Args:
            specialist_name (str): The name of the specialist to call. Supported specialists: "insert_plug", "clear_workspace", "final_failure_recovery", "call_human"

        Returns:
            None
        """

        random_number = random.random()

        if specialist_name == "insert_plug":
            if random_number < 0.5:
                return "insert_plug succeeded"
            else:
                return "insert_plug failed"
        elif specialist_name == "clear_workspace":
            if random_number < 0.5:
                return "clear_workspace succeeded"
            else:
                return "clear_workspace failed"
        elif specialist_name == "final_failure_recovery":
            if random_number < 0.5:
                return "final_failure_recovery succeeded"
            else:
                return "final_failure_recovery failed"
        elif specialist_name == "call_human":
            return "Has informed human, awaiting response"
        else:
            raise ValueError(f"Unsupported specialist: {specialist_name}")


    def build_agent(self):

        self.tools = [self.call_specialist]
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        agent = self.vlm.bind_tools(self.tools)
        return agent


