ifile = '/home/dgyalts1/cs87/Lab01-dgyalts1-mnaagaa1/resultsChervilC.txt'

nt = []
rt = []
with open(ifile, 'r') as file:

    lines = file.readlines()

    for i in range(2, len(lines), 14):
        nt.append(lines[i])
    for i in range(6, len(lines), 14):
        rt.append(lines[i])

for i in range(len(rt)):
    rt[i] = rt[i].strip()
    rt[i] = rt[i].lstrip("Total time: ")
    rt[i] = rt[i].rstrip(" seconds")
    nt[i] = nt[i].strip()
    nt[i] = nt[i] + ", " + rt[i] + "\n"

ofile = 'resultsParserCOutput.txt'

with open(ofile, 'w') as file:

    file.writelines(nt)

# for line in nt:
#     print(line.strip())

# for line in rt:
#     print(line.strip())
