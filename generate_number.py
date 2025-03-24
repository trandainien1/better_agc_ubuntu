with open("idx_ILSVRC2012.csv", "w") as file:
    for i in range(30000, 40000):
            file.write(f"{i}\n")