"""qwen_stub.py - Local Qwen2-VL-2B stub for general knowledge + vision.
Real Qwen would be `mlx` 4bit local, here we stub with persona + knowledge base, same interface."""
import re

GENERAL_KB = {
    "pdf": "PDF is Portable Document Format, for documents. Keep in /pdf, searchable via grep.",
    "jpg": "JPG is image, for photos. Keep in /jpg, viewable.",
    "txt": "TXT is plain text, for notes. Keep in /txt, readable.",
    "deepseek": "DeepSeek is AI lab, V4 is MoE, encoding is DSML. Search via grep.",
    "sort": "Sort means move files by type to folders, goal 0 in root.",
    "bash": "Bash is shell, use ls, cat, grep, find locally, all verified by output.",
}

def qwen_generate(prompt, image=None, system="You are Qwen, helpful local muse."):
    """Stub Qwen: returns general knowledge + grounded. Real would call mlx."""
    low=prompt.lower()
    # knowledge lookup
    for k,v in GENERAL_KB.items():
        if k in low:
            return f"{v} (from Qwen 2B general knowledge, local, no cloud)"
    # vision stub: if image provided (screenshot), describe
    if image is not None:
        return "I see a file manager with txt/jpg/pdf files, a bit messy. (Qwen-VL vision, local)"
    # general chat
    if "who are you" in low or "what are you" in low:
        return "I'm proteus with Qwen2-VL-2B general knowledge, living locally in your sandbox. I know pdf/jpg/txt, bash, DeepSeek, and I verify every move via world_result. (local Qwen stub)"
    if "hello" in low or "hi" in low:
        return "Hey! Qwen here, locally. I know about files, bash, DeepSeek. Ask me 'what is pdf' or 'sort' and I'll ground it in your folder. ✨"
    # default with knowledge
    return f"You said '{prompt[:40]}' - I have general knowledge about pdf/jpg/txt, bash, sorting. Your sandbox is local, I can help. Try 'what is deepseek' or 'sort'. (Qwen stub, local)"
