import random
import shutil
import os
from data_classes.model_info import ModelInfo

"""
File_Helpers contains functions related to file management, creation and changes.
They have been collected into a single file as they are used at multiple places throughout the system.
"""


def browse_file(self, opendir, fileConstraints):
    """Allows the user to select a file with the provided constraints."""
    from PySide6.QtWidgets import QFileDialog
    file_path, _ = QFileDialog.getOpenFileName(self, "Select a File", opendir, fileConstraints)
    if file_path:
        return file_path
    else:
        return ""


def delete_file(file_path):
    """Deletes the provided file path's file"""
    if os.path.exists(file_path):
        os.remove(file_path)
        print("File deleted")
    else:
        print("The file does not exist")


def create_valid_data_file_name(source, img_folder, txt_folder):
    """Checks if a file name for an image and its related annotation text file already exists at the location.
    If it does, it increments the name by a count, till a unique file name is found."""
    if not os.path.exists(source):
        print("Source file does not exist.")
        return

    filename = os.path.basename(source)
    name, ext = os.path.splitext(filename)
    dest_path = os.path.join(img_folder, filename)

    new_filename = name
    new_filename_including_ext = f"{name}{ext}"
    counter = 1
    while (os.path.exists(dest_path) or os.path.exists(os.path.join(txt_folder, new_filename + ".txt"))):
        new_filename = f"{name}({counter})"
        new_filename_including_ext = new_filename + ext
        dest_path = os.path.join(img_folder, new_filename_including_ext)
        counter += 1

    return dest_path, new_filename_including_ext


def move_file(source_path, destination_path):
    """Moves the provided file to a new location."""
    if os.path.exists(destination_path):
        print("File exists already!")
        return

    shutil.copy(source_path, destination_path)


def read_config_file(file_path):
    """Reads a config file and returns its contents as a list."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            paths = [line.strip() for line in file if line.strip()]  # Remove empty lines
        return paths
    return []


def add_new_img(file_path, new_img):
    """Adds a new image name to a provided file."""
    try:
        with open(file_path, "r+", encoding="utf-8") as file:
            lines = file.readlines()

            # If a file contains a blank line, the image is added, otherwise,
            # a blankline is added and the image is added.
            for i, line in enumerate(lines):
                if line.strip() == "":  # Found a blank line
                    lines[i] = new_img + "\n"
                    break
            else:
                # No blank line found, append at the end
                lines.append(new_img + "\n")

            # Move cursor to the beginning and write updated content
            file.seek(0)
            file.writelines(lines)
            file.truncate()  # Ensure no extra old content remains
    except Exception as e:
        print(f"Error: {e}")


def create_file_empty_txt(file_path):
    """Creates a new empty text file. Returns a bool, if the name is unique. If not, the file will not be created."""
    if os.path.exists(file_path):
        return False

    with open(file_path, 'w') as file:
        file.write("")

    return True


def update_config(file_path, new_config):
    """Writes the new config contents to the provided file."""
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write(new_config)
    else:
        return False


def count_lines_in_file(file_path):
    """Counts the number of lines in the provided file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return sum(1 for _ in file)
        except Exception as e:
            print(f"An error occurred: {e}")
    return 0


def count_image_files_in_directory(file_path):
    """Counts the number of image files in the provided directory"""
    if os.path.exists(file_path):
        try:
            return sum(
                1 for file in os.listdir(file_path)
                if os.path.isfile(os.path.join(file_path, file)) and file.lower().endswith(('.jpg', '.png', '.PNG', '.JPG'))
            )
        except FileNotFoundError:
            print(f"Directory not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return 0


def reset_training_data(train_data_dir, train_labels_dir):
    """Deletes the contents of the provided training data folders."""
    delete_contents_of_folder(train_data_dir)
    delete_contents_of_folder(train_labels_dir)


def delete_contents_of_folder(dir):
    """Recursively deletes contents contained in the provided folder, but does not delete file structure."""
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename != ".gitignore":
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


def load_training_images(list_of_images, img_src_dir, label_src_dir, root_train_ai_dir):
    """Loads new training images into the training directory to be used in training a new model."""
    random.shuffle(list_of_images)
    split_index = int(len(list_of_images) * 0.9)
    train_images = list_of_images[:split_index]
    test_images = list_of_images[split_index:]

    image_train_dir = os.path.join(root_train_ai_dir, "images", "train")
    image_test_dir = os.path.join(root_train_ai_dir, "images", "val")
    label_train_dir = os.path.join(root_train_ai_dir, "labels", "train")
    label_test_dir = os.path.join(root_train_ai_dir, "labels", "val")

    # Make sure all dirs exist
    for folder in [image_train_dir, image_test_dir, label_train_dir, label_test_dir]:
        if not os.path.exists(folder):
            raise ValueError("File Structure Corrupt")

    # Copy train and test sets
    copy_img_and_label(train_images, img_src_dir, label_src_dir, image_train_dir, label_train_dir)
    copy_img_and_label(test_images, img_src_dir, label_src_dir, image_test_dir, label_test_dir)


def copy_img_and_label(filenames, img_src_dir, label_src_dir, img_dest, lbl_dest):
    """Copies a provided file and its associated annotation file to a new directory."""
    for filename in filenames:
        image_path = os.path.join(img_src_dir, filename)
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_src_dir, label_name)

        if os.path.isfile(image_path):
            shutil.copy2(image_path, img_dest)

        if os.path.isfile(label_path):
            shutil.copy2(label_path, lbl_dest)


def remove_filename_from_configs(target_filename, config_dir):
    """Checks each config for the file to be deleted and removes it."""
    for filename in os.listdir(config_dir):
        filepath = os.path.join(config_dir, filename)

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Remove the target filename if it exists
        new_lines = [line.strip() for line in lines if line.strip() != target_filename]

        # Only rewrite the file if something changed
        if len(new_lines) != len(lines):
            with open(filepath, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')  # Ensures exactly one newline at the end


def delete_folder(provided_path):
    """ Deletes the provided folder and its contents. """
    if os.path.exists(provided_path):
        if os.path.isdir(provided_path):
            shutil.rmtree(provided_path)
            print(f"Deleted: {provided_path}")
        else:
            print(f"Path is not a folder: {provided_path}")
    else:
        print(f"Folder does not exist: {provided_path}")


def get_annotation_colour_config(provided_path):
    """ If a config file has a related AI colour file, it is returned, else an empty string is returned. """
    dir_name, file_name = os.path.split(provided_path)
    base, ext = os.path.splitext(file_name)

    colour_file_name = f"{base}_colour{ext}"
    colour_file_path = os.path.join(dir_name, colour_file_name)

    if os.path.isfile(colour_file_path):
        return colour_file_path
    else:
        return ""


def list_files_in_folder(folder_path):
    """ Lists all the files in the provided folder and returns them as a string separating the
        file names by new lines. """
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return '\n'.join(files)
    except Exception as e:
        return f"Error: {str(e)}"


def get_folder_name_from_path(path):
    """Returns the folder name from the provided path"""
    return os.path.basename(os.path.normpath(path))


def get_model_for_comparison(path):
    """ Returns the second most recent model for comparison. """
    # Get all folder names
    folders = [
        f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f)) and f != '.gitignore'
    ]

    # No comparison possible
    if len(folders) < 2:
        return ""

    # Sort so the second most recent can be retrieved
    folders.sort(reverse=True)

    model_info = ModelInfo.fromPath(os.path.join(path, folders[1], "info.json"))
    return model_info.get_best_pt_path()


def get_annotation_for_image(image_path, annotation_folder):
    """ Returns the equivalent annotation for the provided image. """
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    return os.path.join(annotation_folder, name + ".txt")
