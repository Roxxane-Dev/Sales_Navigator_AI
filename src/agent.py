import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from tools import query_predictions, semantic_search, get_segment_summary, get_business_kpis

load_dotenv()

available_tools = {
    "query_predictions": query_predictions,
    "semantic_search": semantic_search,
    "get_segment_summary": get_segment_summary,
    "get_business_kpis": get_business_kpis
}

class NINAAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        system_prompt = 
        self.history = [{"role": "system", "content": system_prompt}]
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_predictions",
                    "description": "Filtra clientes usando condiciones SQL WHERE y ORDER BY.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql_where": {"type": "string", "description": "Condición WHERE en SQL. Ej: score_propension > 0.7"},
                            "order_by": {"type": "string", "description": "Condición ORDER BY en SQL. Ej: score_propension DESC"},
                            "limit": {"type": "integer", "description": "Límite de resultados a retornar. Por defecto 20."}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": "Busca perfiles de clientes por lenguaje natural usando similitud semántica. Útil para descripciones abiertas o búsquedas que no se pueden resolver fácilmente con un SQL WHERE exacto.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Descripción del perfil de cliente que se busca en lenguaje natural."},
                            "top_k": {"type": "integer", "description": "Número de clientes a retornar. Por defecto 10."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_segment_summary",
                    "description": "Obtiene un resumen agregado de los 4 segmentos de clientes con la cantidad, score promedio y tasa de compra.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_business_kpis",
                    "description": "Obtiene los KPIs globales del negocio (total clientes, riesgo de churn, conversión global, etc).",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        
        system_msg = self.history[0]
        
        while True:
            recent_history = self.history[1:]
            history_to_send = recent_history[-6:] if len(recent_history) > 6 else recent_history
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[system_msg] + history_to_send,
                tools=self.tools,
                temperature=0.2
            )
            
            message = response.choices[0].message
            
            self.history.append(message)
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        func_args = {}
                        
                    if func_name in available_tools:
                        result = available_tools[func_name](**func_args)
                    else:
                        result = {"error": f"Tool {func_name} not found"}
                        
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            else:
                return message.content
                
    def reset(self):
        self.history = [self.history[0]]

def main():
    print("=========================================================")
    print("Iniciando NINA Agent... (Escribe 'salir' para terminar)")
    print("=========================================================\n")
    agent = NINAAgent()
    
    while True:
        try:
            user_input = input("Tú: ")
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break
            if not user_input.strip():
                continue
            
            response = agent.chat(user_input)
            print(f"\nNINA: {response}\n")
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break
        except Exception as e:
            print(f"\nError de comunicación con la API: {e}\n")

if __name__ == "__main__":
    main()
