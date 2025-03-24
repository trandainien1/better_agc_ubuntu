with open("idx_ILSVRC2012.csv", "w") as file:
    for i in range(40000, 50000):
            file.write(f"{i}\n")