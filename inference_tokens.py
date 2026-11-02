def token_layout(tokenizer, row, answer=True):
    start = tokenizer.convert_tokens_to_ids('<|im_start|>')
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if start == end or start == tokenizer.unk_token_id or end == tokenizer.unk_token_id:
        raise ValueError('Wrong Qwen tokenizer')
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    ids = [start] + enc('system\nYou are a helpful assistant.') + [end] + enc('\n')
    labels = [-100] * len(ids)
    for turn in row['conversations']:
        role = {'human': 'user', 'gpt': 'assistant'}[turn['from']]
        if not answer and role != 'user':
            continue
        header = [start] + enc(role + '\n')
        body = []
        for i, chunk in enumerate(turn['value'].split('<image>')):
            if i:
                body.append(-200)
            body.extend(enc(chunk))
        segment = header + body + [end] + enc('\n')
        ids += segment
        labels += [-100] * len(segment) if role == 'user' else [-100] * len(header) + body + [end] + enc('\n')
    if not answer:
        tail = [start] + enc('assistant\n')
        ids += tail
        labels += [-100] * len(tail)
    if ids.count(-200) != 2 or any((y != -100 for x, y in zip(ids, labels) if x == -200)):
        raise ValueError('Image token masking')
    if answer and (not any((y != -100 for y in labels))):
        raise ValueError('No answer labels')
    if len(ids) + 1458 > 4096:
        raise ValueError('Sequence exceeds 4096; no silent truncation')
    return (ids, labels)
