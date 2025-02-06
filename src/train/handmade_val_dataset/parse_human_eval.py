import json

output = []

with open("human_eval.txt", encoding="utf8") as f:
    while True:
        group_type = f.readline().strip()
        if not group_type:
            break

        for _ in range(5):
            text = f.readline().strip()[3:]
            fen = f.readline().strip()
            output.append(
                {
                    "type": group_type,
                    "text": text,
                    "fen": fen,
                }
            )

        f.readline()


with open("handmade.json", "w", encoding="utf8") as f:
    json.dump(output, f)
