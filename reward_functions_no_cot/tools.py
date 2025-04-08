

def get_last_line(text: str) -> str:
    lines = text.split('\n')
    last_line = '\n'.join(lines[-1:])
    
    return last_line