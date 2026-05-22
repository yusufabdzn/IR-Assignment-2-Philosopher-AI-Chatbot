from tools.web_search import web_search

class ToolBox:
    def __init__(self, api_key):
        self.api_key = api_key

    def run(self, tool_name, query):
        if tool_name == "web_search":
            return web_search(query, self.api_key)

        raise ValueError(f"Unknown tool: {tool_name}")