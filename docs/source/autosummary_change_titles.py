import os

gen_folder = os.path.join(os.getcwd(), "docs", "source", "generated")

# For files in the folder and subfolders
for root, dirs, files in os.walk(gen_folder):
    for filename in files:
        if filename.endswith(".rst"):
            file_path = os.path.join(root, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Get the title from the first line
            # module1.module2.class_name.method_name -> method_name
            if lines:
                first_line = lines[0].strip()
                first_line_parts = first_line.split(".")
                new_title = first_line_parts[-1] if first_line_parts else first_line
                lines[0] = new_title + "\n"
                lines[1] = "=" * len(new_title) + "\n"

            with open(file_path, "w") as file:
                file.writelines(lines)