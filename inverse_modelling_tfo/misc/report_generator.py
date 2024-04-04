"""
Generate a report as a markdown file for future reference. Useful for storing expreiment results.
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
from mdutils.mdutils import MdUtils
from matplotlib.figure import Figure


class MarkdownReport:
    """
    Create a Markdown report with text and image sections.

    Example Usage:
    -------------
    ```python
    from report_generator import MarkdownReport
    from pathlib import Path
    import matplotlib.pyplot as plt

    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    fig = plt.gcf()
    report = MarkdownReport(save_directory=Path("reports"), file_name="example_report", title="Example Report")
    report.add_text_report("Heading 1", "This is a text section.")
    report.add_image_report("Heading 2", fig)
    report.add_code_report("Heading 3", "print('Hello World!')")
    report.save_report()    # Saves to reports/example_report.md
    ```
    """

    FIG_FOLDER = "figures"

    def __init__(self, save_directory: Path, file_name: str, title: str, dpi: int = 300):
        """
        Args:
            save_directory: Base directory to save the markdown file along with any related images.
            file_name: Name of the markdown file (without the .md extension)
            title: Title that goes at the top of the markdown file.
            dpi: DPI of the images to be saved in the report. Default is 300.
        """
        self.save_directory = save_directory
        self.file_name = file_name
        self.title = title
        self.dpi = dpi
        markdown_save_path = save_directory / f"{file_name}.md"
        self.md_file = MdUtils(str(markdown_save_path), title=title, author="Rishad Joarder")
        self.sections: List[Section] = []

    def add_text_report(self, heading, text) -> None:
        """
        Add a text section to the report.
        """
        self.sections.append(TextSection(heading, text))

    def add_image_report(self, heading, matplotlib_figure: Figure) -> None:
        """
        Add an image section to the report.
        """
        # Create a unique name and save the image
        image_save_path = self.save_directory / MarkdownReport.FIG_FOLDER / f"{self.file_name}_{len(self.sections)}.png"
        if image_save_path.parent.exists() is False:  # Create the figures folder if it doesn't exist
            self._create_figures_folder()
        matplotlib_figure.savefig(str(image_save_path), dpi=self.dpi)
        # The image link in the markdown file should be relative to the markdown file
        image_section = ImageSection(heading, str(image_save_path.relative_to(self.save_directory)))
        self.sections.append(image_section)

    def add_code_report(self, heading, code, language="") -> None:
        """
        Add a code section to the report.
        """
        code_section = CodeSection(heading, code, language)
        self.sections.append(code_section)

    def save_report(self):
        """
        Render and Save the report
        """
        for section in self.sections:
            if section.is_valid():
                section.render(self.md_file)

        self.md_file.create_md_file()

    def _create_figures_folder(self):
        """
        Create a folder to store the figures.
        """
        figures_folder = self.save_directory / MarkdownReport.FIG_FOLDER
        figures_folder.mkdir(exist_ok=True)


class Section(ABC):
    """
    Abstract class for a section in the markdown report. All types of sections must inherit from this class.
    """

    @abstractmethod
    def render(self, md_file: MdUtils):
        """
        Render the section in the markdown file. This section should include all the pre-processing required to render 
        followed by a call to some MdUtils methods to render the section.
        
        Example MdUtils methods: new_header, new_paragraph, new_inline_image, insert_code, etc.
        """

    @abstractmethod
    def is_valid(self) -> bool:
        """
        Is the section valid to be rendered in the markdown file. If this returns False, the section will be ignored
        during rendering.
        """


class TextSection(Section):
    """
    Single text section in the markdown report.
    """

    def __init__(self, heading, text):
        self.heading = heading
        self.text = text

    def render(self, md_file: MdUtils):
        md_file.new_header(level=1, title=self.heading)
        md_file.new_paragraph(self.text)
        md_file.new_line()

    def is_valid(self) -> bool:
        if len(self.text) == 0:
            return False
        return True


class ImageSection(Section):
    """
    Single image section in the markdown report.
    """

    def __init__(self, heading, image_path: str):
        self.heading = heading
        self.image_path = image_path

    def render(self, md_file: MdUtils):
        md_file.new_header(level=1, title=self.heading)
        md_file.new_line()
        # mdFile.new_inline_image(text=self.heading, path=str(self.image_path))
        md_file.new_line(text=rf"![{self.heading}]({self.image_path})")
        md_file.new_line()

    def is_valid(self) -> bool:
        return True


class CodeSection(Section):
    """
    Single code section in the markdown report.
    """

    def __init__(self, heading: str, code: str, language: str = ""):
        self.heading = heading
        self.code = code
        self.language = language

    def render(self, md_file: MdUtils):
        md_file.new_header(level=1, title=self.heading)
        md_file.insert_code(self.code, language=self.language)
        md_file.new_line()

    def is_valid(self) -> bool:
        if len(self.code) == 0:
            return False
        return True
