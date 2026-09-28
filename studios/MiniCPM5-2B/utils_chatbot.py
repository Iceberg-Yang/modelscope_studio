def organize_messages(message, history=None):
    """Build chat messages from history tuples [[user, assistant], ...]."""
    msg_ls = [{"role": "system", "content": "You are a helpful assistant."}]
    if history:
        for turn in history:
            if not turn:
                continue
            user_text = turn[0] if len(turn) > 0 else None
            assistant_text = turn[1] if len(turn) > 1 else None
            if user_text:
                msg_ls.append({"role": "user", "content": user_text})
            if assistant_text:
                msg_ls.append({"role": "assistant", "content": assistant_text})
    msg_ls.append({"role": "user", "content": message})
    return msg_ls
