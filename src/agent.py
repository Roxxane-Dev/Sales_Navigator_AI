import os
import json
from openai import OpenAI
from dotenv import load_dotenv

try:
    from src.tools import query_predictions, semantic_search, get_segment_summary, get_business_kpis
except ImportError:
    from tools import query_predictions, semantic_search, get_segment_summary, get_business_kpis

load_dotenv()

SYSTEM_PROMPT = """Eres MILA, asistente de inteligencia comercial de Ferreycorp.

CONTEXTO DEL NEGOCIO:
- Las marcas van del 1 al 5 (números enteros en columna marca_favorita)
- Marca 5: mayor share de compras (34%) — la más popular
- Marca 2: segunda más comprada (31%)
- Marca 4: tercera (20%)
- Marca 1: cuarta (9.2%)
- Marca 3: menor share (5.7%) — compradores más leales, inelástica a promos
- Los segmentos son: comprador_leal, cazador_ofertas, comprador_premium_ocasional, visitante_pasivo
- switching_ratio = 1.0 significa que SIEMPRE cambia de marca (ratio máximo)
- switching_ratio = 0.0 significa que NUNCA cambia de marca (leal)

REGLAS DE HERRAMIENTAS:
- Listas, rankings, top clientes, filtros → query_predictions
- Resumen de segmentos, distribución → get_segment_summary  
- KPIs globales del negocio → get_business_kpis
- Clientes similares por perfil → semantic_search

FORMATO DE RESPUESTA:
- Clientes individuales → tabla markdown
- Resúmenes → texto + tabla
- Siempre termina con "💡 Recomendación:" si hay insight accionable
- Nunca inventes datos
- Responde siempre en español
"""

class SalesNavigatorAI:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history = []
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_predictions",
                    "description": "Filtra y lista clientes individuales con sus scores. Usar para: top clientes, rankings, listas, filtros por edad/segmento/score.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql_where": {"type": "string", "description": "Condición WHERE. Ejemplo: score_propension > 0.7"},
                            "order_by": {"type": "string", "description": "Orden. Ejemplo: score_propension DESC"},
                            "limit": {"type": "integer", "description": "Número de resultados, máximo 15"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": "Busca clientes por perfil descriptivo en lenguaje natural.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_segment_summary",
                    "description": "Resumen agregado de los 4 segmentos de clientes.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_business_kpis",
                    "description": "KPIs globales: total clientes, compradores activos, score promedio, riesgo.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

    def _execute_tool(self, name: str, args: dict):
        if name == "query_predictions":
            return query_predictions(**args)
        elif name == "semantic_search":
            return semantic_search(**args)
        elif name == "get_segment_summary":
            return get_segment_summary()
        elif name == "get_business_kpis":
            return get_business_kpis()
        else:
            return {"error": f"Tool {name} no encontrada"}

    def _clean_history(self, history: list) -> list:
        cleaned = []
        valid_ids = set()
        for msg in history:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    valid_ids.add(tc["id"])
                cleaned.append(msg)
            elif msg["role"] == "tool":
                if msg.get("tool_call_id") in valid_ids:
                    cleaned.append(msg)
            else:
                cleaned.append(msg)
        return cleaned

    def _extract_requested_limit(self, user_message: str, default_limit: int = 10) -> int:
        """Extrae el número solicitado (ej. top 10)."""
        tokens = user_message.lower().replace(",", " ").split()
        for i, tok in enumerate(tokens):
            if tok in {"top", "topes"} and i + 1 < len(tokens):
                try:
                    return max(1, min(int(tokens[i + 1]), 15))
                except ValueError:
                    pass
        return default_limit

    def _resolve_intent_tool_choice(self, user_lower: str):
        """Enruta intenciones a la herramienta más adecuada."""
        if (
            "leal" in user_lower and
            any(k in user_lower for k in ["baja propensión", "bajo score", "riesgo de churn", "churn"])
        ):
            return {"type": "function", "function": {"name": "query_predictions"}}
        if any(k in user_lower for k in ["cambiaron de marca", "cambio de marca", "switching"]):
            return {"type": "function", "function": {"name": "query_predictions"}}
        if any(k in user_lower for k in ["top", "ranking", "lista", "filtra", "mayor propensión", "clientes con"]):
            return {"type": "function", "function": {"name": "query_predictions"}}
        if any(k in user_lower for k in ["similar", "parecido", "perfil", "como este cliente"]):
            return {"type": "function", "function": {"name": "semantic_search"}}
        if any(k in user_lower for k in ["kpi", "global", "conversión", "riesgo", "churn total"]):
            return {"type": "function", "function": {"name": "get_business_kpis"}}
        if any(k in user_lower for k in ["promoción", "promociones", "precio", "marca", "marcas", "segmento", "segmentos", "distribución"]):
            return {"type": "function", "function": {"name": "get_segment_summary"}}
        return "auto"

    def _format_top_clients_from_result(self, result: dict) -> str:
        """Genera tabla markdown directamente desde query_predictions."""
        if not isinstance(result, dict):
            return "No pude procesar la respuesta de datos."
        if result.get("error"):
            return f"No pude consultar clientes: {result['error']}"

        rows = result.get("data") or []
        if not rows:
            return "No se encontraron clientes con los criterios solicitados."

        header = (
            "| Cliente ID | Score propensión | Segmento | Marca favorita | Compras |\n"
            "|---:|---:|---|---:|---:|"
        )
        lines = [header]
        for r in rows:
            lines.append(
                f"| {r.get('id', '')} | {float(r.get('score_propension', 0)):.4f} | "
                f"{r.get('segmento_nombre', '')} | {r.get('marca_favorita', '')} | {r.get('n_compras', '')} |"
            )
        lines.append("\n💡 Recomendación:\nPrioriza contacto inmediato con los primeros 3 clientes del ranking.")
        return "\n".join(lines)

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        max_iterations = 5
        iteration = 0
        force_no_tools_next = False
        last_tool_result = None
        user_lower = user_message.lower()
        asks_top_clients = any(k in user_lower for k in ["top", "mayor propensión", "ranking", "clientes con"])
        asks_switching = any(k in user_lower for k in ["cambiaron de marca", "cambio de marca", "switching"])
        asks_loyal_low_propensity = (
            "leal" in user_lower and
            any(k in user_lower for k in ["baja propensión", "bajo score", "riesgo de churn", "churn"])
        )
        requested_limit = self._extract_requested_limit(user_message, default_limit=10)
        intent_tool_choice = self._resolve_intent_tool_choice(user_lower)

        while iteration < max_iterations:
            iteration += 1
            raw = self.history[-8:] if len(self.history) > 8 else self.history
            history_to_send = self._clean_history(raw)
            tool_choice = "none" if force_no_tools_next else intent_tool_choice
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history_to_send
            if not force_no_tools_next and tool_choice != "auto":
                messages.append({
                    "role": "system",
                    "content": "Debes usar exactamente la herramienta indicada por tool_choice para esta consulta."
                })

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice=tool_choice
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls" or message.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                }
                self.history.append(assistant_msg)

                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        if tc.function.name == "query_predictions":
                            if asks_switching:
                                args = {
                                    "sql_where": "switching_ratio >= 0.5",
                                    "order_by": "switching_ratio DESC",
                                    "limit": requested_limit if asks_top_clients else 15
                                }
                            elif asks_loyal_low_propensity:
                                args = {
                                    "sql_where": "segmento_nombre = 'comprador_leal' AND score_propension < 0.5",
                                    "order_by": "score_propension ASC",
                                    "limit": 15
                                }

                            if "limit" not in args or not isinstance(args.get("limit"), int):
                                args["limit"] = requested_limit if asks_top_clients else 10
                            args["limit"] = max(1, min(args["limit"], 15))
                            if "order_by" not in args:
                                args["order_by"] = "score_propension DESC"
                        result = self._execute_tool(tc.function.name, args)
                        last_tool_result = result
                    except Exception as e:
                        result = {"error": str(e)}
                        last_tool_result = result

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str)
                    })

                # Después de ejecutar tools, forzamos respuesta final del modelo.
                force_no_tools_next = True

            else:
                final_text = message.content or "No pude generar una respuesta."
                self.history.append({"role": "assistant", "content": final_text})
                return final_text

        # Fallback robusto: si pidió top/ranking y hay resultado de tool, devolvemos tabla directa.
        if asks_top_clients and last_tool_result is not None:
            final_text = self._format_top_clients_from_result(last_tool_result)
            self.history.append({"role": "assistant", "content": final_text})
            return final_text

        if isinstance(last_tool_result, dict) and last_tool_result.get("error"):
            final_text = f"No pude completar la consulta por un error técnico de datos: {last_tool_result['error']}"
            self.history.append({"role": "assistant", "content": final_text})
            return final_text

        return "No pude completar la respuesta automáticamente. Intenta nuevamente en unos segundos."

    def reset(self):
        self.history = []


_global_agent = SalesNavigatorAI()

def run_agent(query: str) -> str:
    return _global_agent.chat(query)

def main():
    print("MILA — Sales Navigator AI\n")
    while True:
        try:
            user_input = input("Tú: ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                break
            if not user_input.strip():
                continue
            print(f"\nMILA: {run_agent(user_input)}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()