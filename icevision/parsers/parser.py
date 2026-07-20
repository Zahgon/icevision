__all__ = ["ParserInterface", "Parser"]

from icevision.imports import *
from icevision.utils import *
from icevision.utils.code_template import *
from icevision.core import *
from icevision.data import *


def camel_to_snake(name):
    pass


class ParserInterface(ABC):
    @abstractmethod
    def parse(
        self, data_splitter: DataSplitter, autofix: bool = True, show_pbar: bool = True
    ) -> List[List[RecordType]]:
        pass


class Parser(ParserInterface, ABC):

    def __init__(
        self,
        template_record,
        class_map: Optional[ClassMap] = None,
        idmap: Optional[IDMap] = None,
    ):
        self.template_record = template_record
        self.idmap = idmap or IDMap()

    @abstractmethod
    def __iter__(self) -> Any:
        pass

    @abstractmethod
    def parse_fields(self, o, record: BaseRecord, is_new: bool) -> None:
        pass

    def create_record(self) -> BaseRecord:
        pass

    def prepare(self, o):
        pass

    def parse_dicted(self, show_pbar: bool = True) -> Dict[int, RecordType]:
        pass

    def _check_path(self, path: Union[str, Path] = None):
        pass

    def parse(
        self,
        data_splitter: DataSplitter = None,
        autofix: bool = True,
        show_pbar: bool = True,
        cache_filepath: Union[str, Path] = None,
    ) -> List[List[BaseRecord]]:
        pass

    @classmethod
    def _templates(cls) -> List[str]:
        pass

    @classmethod
    def generate_template(cls, record):
        pass
