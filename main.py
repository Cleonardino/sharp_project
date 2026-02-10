"""
This is the SHARP Project main entry point, containing a menu that let the user chose which part of
the project is executed
"""

import config.config as cf
from data_pipeline.src.import_data import download_dataset
from data_pipeline.src.validate_extract_data import DatasetValidatorExtractor

def display_menu():
    """Display all the options available"""
    print("\n" + "="*40)
    print("OPTIONS AVAILABLE")
    print("="*40)
    print("1. Download Dataset")
    print("2. Validate Dataset")
    print("5. Exit")
    print("="*40)

def main():
    """Main Entry Point. Get user input to what part of the project needs to be executed"""
    while True:
        display_menu()

        choice = input("\nEnter your choice (1-5): ").strip()

        # Execute action based on choice
        if choice == "1":
            download_dataset(
                cf.IMAGE_DIR,
                cf.ANNOTATION_ZIP_DIR
            )
        elif choice == "2":
            DatasetValidatorExtractor(
                cf.IMAGE_DIR,
                cf.ANNOTATION_ZIP_DIR
            ).validate()
        elif choice == "5":
            print("\nExiting now")
            break
        else:
            print("\nInvalid choice! Please enter a number between 1 and 5.")

        # Pause before showing menu again
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
