from pathlib import Path
from praatio import textgrid

TEXTGRID_PATH = Path(r"data/dataset/opencpop/textgrids/2002.TextGrid")

tg = textgrid.openTextgrid(str(TEXTGRID_PATH), includeEmptyIntervals=False)

print("TextGrid:", TEXTGRID_PATH)
print("Tier names:")
print(tg.tierNames)

for tier_name in tg.tierNames:
    tier = tg.getTier(tier_name)
    print("\n==============================")
    print("Tier:", tier_name)
    print("Entry count:", len(tier.entries))
    print("First 20 entries:")

    for entry in tier.entries[:20]:
        start, end, label = entry
        print(f"{start:.3f} - {end:.3f} : {label}")