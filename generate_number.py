with open("output.txt", "w") as file:
    for i in range(5000, 10000):
        file.write(f"{i}\n")