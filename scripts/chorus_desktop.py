#!/usr/bin/env python3
"""
CHORUS Desktop - A GUI application for testing the CHORUS RAG system.

Features:
- Chat interface with Claude AI
- RAG-powered knowledge base queries
- Conversation history
- Status indicators for RAG server connection

Usage:
    python chorus_desktop.py

Requires:
    - ANTHROPIC_API_KEY environment variable
    - RAG HTTP server running on localhost:8765 (optional but recommended)
"""

import json
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import urllib.request
import urllib.parse
from datetime import datetime

try:
    import anthropic
except ImportError:
    messagebox.showerror("Missing Dependency", "Please install anthropic: pip install anthropic")
    raise SystemExit(1)


RAG_SERVER_URL = "http://localhost:8765"


class ChorusDesktop:
    def __init__(self, root):
        self.root = root
        self.root.title("CHORUS Desktop")
        self.root.geometry("900x700")
        self.root.minsize(600, 400)

        # State
        self.client = None
        self.rag_connected = False
        self.conversation_history = []
        self.is_processing = False

        # Configure styles
        self.setup_styles()

        # Build UI
        self.build_ui()

        # Initialize
        self.check_connections()

    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.configure("Status.TLabel", padding=5)
        style.configure("Connected.TLabel", foreground="green")
        style.configure("Disconnected.TLabel", foreground="red")
        style.configure("Send.TButton", padding=(20, 10))

    def build_ui(self):
        """Build the main UI."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: Status bar
        self.build_status_bar(main_frame)

        # Middle: Chat area
        self.build_chat_area(main_frame)

        # Bottom: Input area
        self.build_input_area(main_frame)

    def build_status_bar(self, parent):
        """Build the status bar at the top."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        title_label = ttk.Label(status_frame, text="CHORUS", font=("Helvetica", 16, "bold"))
        title_label.pack(side=tk.LEFT)

        subtitle = ttk.Label(status_frame, text="  memory, matchmaker, muse", font=("Helvetica", 10, "italic"))
        subtitle.pack(side=tk.LEFT)

        # Status indicators (right side)
        indicators_frame = ttk.Frame(status_frame)
        indicators_frame.pack(side=tk.RIGHT)

        # RAG status
        ttk.Label(indicators_frame, text="RAG:").pack(side=tk.LEFT, padx=(0, 5))
        self.rag_status_label = ttk.Label(indicators_frame, text="Checking...", style="Status.TLabel")
        self.rag_status_label.pack(side=tk.LEFT, padx=(0, 15))

        # API status
        ttk.Label(indicators_frame, text="Claude API:").pack(side=tk.LEFT, padx=(0, 5))
        self.api_status_label = ttk.Label(indicators_frame, text="Checking...", style="Status.TLabel")
        self.api_status_label.pack(side=tk.LEFT)

        # Refresh button
        refresh_btn = ttk.Button(indicators_frame, text="↻", width=3, command=self.check_connections)
        refresh_btn.pack(side=tk.LEFT, padx=(10, 0))

    def build_chat_area(self, parent):
        """Build the main chat display area."""
        chat_frame = ttk.Frame(parent)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            state=tk.DISABLED,
            padx=10,
            pady=10,
            bg="#fafafa"
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure tags for formatting
        self.chat_display.tag_configure("user", foreground="#0066cc", font=("Helvetica", 11, "bold"))
        self.chat_display.tag_configure("assistant", foreground="#006633", font=("Helvetica", 11, "bold"))
        self.chat_display.tag_configure("system", foreground="#666666", font=("Helvetica", 10, "italic"))
        self.chat_display.tag_configure("tool", foreground="#996600", font=("Helvetica", 10))
        self.chat_display.tag_configure("error", foreground="#cc0000")
        self.chat_display.tag_configure("timestamp", foreground="#999999", font=("Helvetica", 9))

    def build_input_area(self, parent):
        """Build the message input area."""
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X)

        # Text input
        self.message_input = tk.Text(
            input_frame,
            height=3,
            font=("Helvetica", 11),
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        self.message_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.message_input.bind("<Return>", self.on_enter_pressed)
        self.message_input.bind("<Shift-Return>", lambda e: None)  # Allow shift+enter for newlines

        # Buttons frame
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Send button
        self.send_button = ttk.Button(
            buttons_frame,
            text="Send",
            style="Send.TButton",
            command=self.send_message
        )
        self.send_button.pack(fill=tk.X, pady=(0, 5))

        # Clear button
        clear_button = ttk.Button(
            buttons_frame,
            text="Clear Chat",
            command=self.clear_chat
        )
        clear_button.pack(fill=tk.X)

        # Processing indicator
        self.progress_var = tk.StringVar(value="")
        self.progress_label = ttk.Label(input_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.BOTTOM, fill=tk.X)

    def check_connections(self):
        """Check RAG server and API connections."""
        def check():
            # Check RAG server
            try:
                health = self.call_rag("/health")
                if "error" not in health:
                    self.rag_connected = True
                    chunks = health.get("chunks", 0)
                    self.root.after(0, lambda: self.update_rag_status(True, f"Connected ({chunks} chunks)"))
                else:
                    self.rag_connected = False
                    self.root.after(0, lambda: self.update_rag_status(False, "Disconnected"))
            except Exception:
                self.rag_connected = False
                self.root.after(0, lambda: self.update_rag_status(False, "Disconnected"))

            # Check API key
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    self.root.after(0, lambda: self.update_api_status(True, "Ready"))
                except Exception as e:
                    self.root.after(0, lambda: self.update_api_status(False, f"Error: {e}"))
            else:
                self.root.after(0, lambda: self.update_api_status(False, "No API key"))

        threading.Thread(target=check, daemon=True).start()

    def update_rag_status(self, connected, text):
        """Update RAG status indicator."""
        self.rag_status_label.config(text=text)
        if connected:
            self.rag_status_label.config(foreground="green")
        else:
            self.rag_status_label.config(foreground="red")

    def update_api_status(self, connected, text):
        """Update API status indicator."""
        self.api_status_label.config(text=text)
        if connected:
            self.api_status_label.config(foreground="green")
        else:
            self.api_status_label.config(foreground="red")

    def call_rag(self, endpoint: str, params: dict = None) -> dict:
        """Call the RAG HTTP API."""
        url = f"{RAG_SERVER_URL}{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            return {"error": str(e)}

    def on_enter_pressed(self, event):
        """Handle Enter key press."""
        if not event.state & 0x1:  # Not Shift+Enter
            self.send_message()
            return "break"

    def send_message(self):
        """Send the current message to Claude."""
        if self.is_processing:
            return

        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            return

        if not self.client:
            self.append_message("System", "Please set ANTHROPIC_API_KEY environment variable and restart.", "error")
            return

        # Clear input
        self.message_input.delete("1.0", tk.END)

        # Display user message
        self.append_message("You", message, "user")

        # Process in background
        self.is_processing = True
        self.send_button.config(state=tk.DISABLED)
        self.progress_var.set("Processing...")

        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()

    def process_message(self, user_message: str):
        """Process message with Claude (runs in background thread)."""
        try:
            response = self.chat_with_claude(user_message)
            self.root.after(0, lambda: self.append_message("Claude", response, "assistant"))
        except Exception as e:
            self.root.after(0, lambda: self.append_message("Error", str(e), "error"))
        finally:
            self.root.after(0, self.finish_processing)

    def finish_processing(self):
        """Reset UI after processing."""
        self.is_processing = False
        self.send_button.config(state=tk.NORMAL)
        self.progress_var.set("")
        self.message_input.focus()

    def chat_with_claude(self, user_message: str) -> str:
        """Send a message to Claude with RAG tools."""
        system = """You are a helpful assistant with access to the CHORUS knowledge base,
which contains documents about research projects, grants, workshops, and team members
at the Knowledge Lab (University of Chicago).

When answering questions:
1. Use get_context_for_question to retrieve relevant information first
2. Base your answers on the retrieved context
3. Cite your sources
4. If the context doesn't have enough info, say so

Always search the knowledge base before answering questions about the lab."""

        # Tool definitions
        tools = [
            {
                "name": "search_knowledge_base",
                "description": "Search the CHORUS knowledge base for relevant documents about research, grants, workshops, and team members at Knowledge Lab.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results (default 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_context_for_question",
                "description": "Get relevant context from the knowledge base to help answer a question. Use this before answering questions about research, grants, workshops, or team members.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to find context for"
                        }
                    },
                    "required": ["question"]
                }
            }
        ]

        # Build conversation
        self.conversation_history.append({"role": "user", "content": user_message})
        messages = self.conversation_history.copy()

        # Only include tools if RAG is connected
        tools_to_use = tools if self.rag_connected else []

        # Call Claude
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system,
            tools=tools_to_use if tools_to_use else None,
            messages=messages
        )

        # Handle tool use loop
        while response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    # Update UI with tool usage
                    self.root.after(0, lambda n=block.name: self.append_tool_use(n))
                    result = self.handle_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Continue with tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system,
                tools=tools_to_use if tools_to_use else None,
                messages=messages
            )

        # Extract text response
        result = ""
        for block in response.content:
            if hasattr(block, 'text'):
                result += block.text

        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": result})

        return result

    def handle_tool(self, name: str, inputs: dict) -> str:
        """Execute a tool and return result."""
        if name == "search_knowledge_base":
            result = self.call_rag("/search", {
                "q": inputs["query"],
                "top_k": inputs.get("num_results", 5)
            })
            return json.dumps(result, indent=2)

        elif name == "get_context_for_question":
            result = self.call_rag("/context", {"q": inputs["question"]})
            return json.dumps(result, indent=2)

        return json.dumps({"error": f"Unknown tool: {name}"})

    def append_message(self, sender: str, message: str, tag: str):
        """Append a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")

        # Sender
        self.chat_display.insert(tk.END, f"{sender}: ", tag)

        # Message
        self.chat_display.insert(tk.END, f"{message}\n\n")

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_tool_use(self, tool_name: str):
        """Append a tool use indicator."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"  → Using tool: {tool_name}\n", "tool")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def clear_chat(self):
        """Clear the chat display and history."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.conversation_history = []
        self.append_message("System", "Chat cleared. Conversation history reset.", "system")


def main():
    # Check for API key early
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Set it with:")
        print("  export ANTHROPIC_API_KEY=your-key-here")

    root = tk.Tk()
    app = ChorusDesktop(root)
    root.mainloop()


if __name__ == "__main__":
    main()
