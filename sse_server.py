import json
import asyncio
import logging
import uuid
from aiohttp import web
import refine_script

# Configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-load patterns and embeddings from refine_script.
patterns = refine_script.load_patterns()
embeddings = refine_script.compute_category_embeddings(patterns)

# Dictionary mapping client_id to SSE response objects.
clients = {}

async def handle_mcp(request: web.Request) -> web.StreamResponse:
    """
    Unified MCP handler supporting both GET (SSE connection) and POST (JSON-RPC commands).
    
    For GET:
      - Establishes a long-lived SSE connection.
      - Immediately sends an 'initialize' event that includes a generated clientId 
        and indicates that completion is supported.
      - Keeps the connection alive by sending periodic ping events.
    
    For POST:
      - Reads the JSON-RPC request and determines the MCP method.
      - Uses the X-Client-ID header (or the only connected client if unambiguous)
        to locate the SSE connection.
      - For a "complete" method, processes the text using refine_script logic and 
        pushes a completion result onto the SSE channel.
      - Also returns the JSON-RPC response via the HTTP response.
    """
    if request.method == "GET":
        # Establish SSE connection.
        client_id = str(uuid.uuid4())
        logger.info("New SSE connection established, client_id: %s", client_id)
        
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        await response.prepare(request)
        
        # Register the SSE connection.
        clients[client_id] = response
        
        # Immediately send an 'initialize' event.
        init_event = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "capabilities": {
                    "completion": True,
                    "sampling": True,
                    "streaming": True,
                    "context": True
                },
                "clientId": client_id
            }
        }
        event_payload = f"data: {json.dumps(init_event)}\n\n"
        await response.write(event_payload.encode())
        await response.drain()
        
        # Keep the connection alive with periodic pings.
        try:
            while True:
                await asyncio.sleep(15)
                await response.write(b": ping\n\n")
                await response.drain()
        except asyncio.CancelledError:
            logger.info("SSE connection cancelled for client_id: %s", client_id)
        except Exception as e:
            logger.error("Error on SSE connection (client_id %s): %s", client_id, str(e))
        finally:
            clients.pop(client_id, None)
        return response

    elif request.method == "POST":
        # Process the JSON-RPC command.
        try:
            data = await request.json()
        except Exception as e:
            logger.error("Failed to parse JSON: %s", str(e))
            return web.json_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error"
                }
            })
        
        logger.info("Received MCP POST request: %s", data)
        method = data.get("method")
        req_id = data.get("id")
        params = data.get("params", {})
        
        # Look up the SSE connection using header or fallback.
        client_id = request.headers.get("X-Client-ID")
        if not client_id:
            if len(clients) == 1:
                client_id = next(iter(clients))
                logger.info("No X-Client-ID provided; using the only connected client: %s", client_id)
            else:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32000,
                        "message": "Client not connected (ambiguous client selection)"
                    }
                }
                return web.json_response(error_resp)
        
        if client_id not in clients:
            error_resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": "Client not connected"
                }
            }
            return web.json_response(error_resp)
        
        sse_response = clients[client_id]
        
        # Process MCP methods.
        if method == "initialize":
            # Even if the client sends an 'initialize' request, respond with capabilities.
            result = {
                "capabilities": {
                    "completion": True,
                    "sampling": True,
                    "streaming": True,
                    "context": True
                }
            }
            resp_obj = {"jsonrpc": "2.0", "id": req_id, "result": result}
        elif method == "sampling/createMessage":
            logger.info("Received sampling request with params: %s", params)
            text = params.get("text", "")
            if not text.strip():
                logger.info("Empty text received, returning None")
                resp_obj = {"jsonrpc": "2.0", "id": req_id, "result": None}
            else:
                try:
                    # Use refine_script to process text.
                    top_categories = refine_script.find_top_patterns(text, embeddings)
                    prompt = refine_script.generate_prompt(text, top_categories, patterns)
                    refined_text = refine_script.refine_text_with_gpt4(prompt)
                    result = {
                        "items": [{
                            "label": refined_text,
                            "kind": 1,          # Adjust this value if needed.
                            "detail": "Refined text"
                        }]
                    }
                    resp_obj = {"jsonrpc": "2.0", "id": req_id, "result": result}
                except Exception as e:
                    logger.error("Error during text refinement: %s", str(e))
                    resp_obj = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32000,
                            "message": f"Refinement error: {str(e)}"
                        }
                    }
        else:
            resp_obj = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        
        # Send the response on the SSE stream.
        try:
            sse_payload = f"data: {json.dumps(resp_obj)}\n\n"
            await sse_response.write(sse_payload.encode())
            await sse_response.drain()
        except Exception as e:
            logger.error("Failed to send SSE message to client %s: %s", client_id, str(e))
        
        # Also return the JSON-RPC response via HTTP.
        return web.json_response(resp_obj)

    else:
        # Method not allowed.
        return web.Response(status=405, text="Method Not Allowed")

def create_app() -> web.Application:
    """Create the web application with a unified MCP endpoint supporting both GET and POST."""
    app = web.Application()
    # Register the unified route for all methods.
    app.router.add_route("*", "/sse", handle_mcp)
    return app

def main():
    """Main entry point to run the MCP server."""
    app = create_app()
    # Listen on the desired host and port.
    web.run_app(app, host="192.168.11.3", port=8000, access_log=logger)

if __name__ == "__main__":
    main()
