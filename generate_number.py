with open("idx_ILSVRC2012.csv", "w") as file:
    for i in range(10000, 20000):
            file.write(f"{i}\n")