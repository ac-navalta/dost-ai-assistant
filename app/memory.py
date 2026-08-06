class ConversationMemory:

    def __init__(self, max_turns=3):
        self.max_messages = max_turns * 2
        self.messages = []

    def add(self, role, message):

        self.messages.append({
            "role": role,
            "content": message
        })

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages.clear()