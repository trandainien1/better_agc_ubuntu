with open("idx_ILSVRC2012.csv", "w") as file:
    for i in range(20000, 30000):
            file.write(f"{i}\n")