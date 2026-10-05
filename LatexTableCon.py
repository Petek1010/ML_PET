# Convert TSV (tab separated values) to LaTeX style (with &)
input_file = "LatexTableResults1.txt"
output_file = "LatexTableResultsOutput.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

converted = []
for line in lines:
    # Strip newline, replace tabs with &
    line = line.strip().replace("\t", " & ")
    converted.append(line)

# Join rows with LaTeX row ending
converted = [row + r" \\" for row in converted]

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(converted))

print(f"Converted table saved to {output_file}")