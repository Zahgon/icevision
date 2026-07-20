from __future__ import annotations

__all__ = ["RecordCollection"]

from icevision.imports import *
from icevision.utils import *
from icevision.core.record import BaseRecord, autofix_records
from icevision.data.data_splitter import DataSplitter


class RecordCollection:
    def __init__(self, create_record_fn):
        self.create_record_fn = create_record_fn
        self._records = IndexableDict()

    def get_by_record_id(self, record_id):
        pass

    def new(self, records: Sequence[BaseRecord]) -> RecordCollection:
        new = type(self)(self.create_record_fn)
        new._records = IndexableDict([(record.record_id, record) for record in records])
        return new

    def __add__(self, other: RecordCollection) -> RecordCollection:
        return self.new([*self._records.values(), *other._records.values()])

    def make_splits(self, data_splitter: DataSplitter) -> List[RecordCollection]:
        pass

    def autofix(self, show_pbar: int = True):
        records = autofix_records(self._records.values())
        return self.new(records)

    def __getitem__(self, i: Union[int, slice]) -> Union[BaseRecord, RecordCollection]:
        if isinstance(i, slice):
            return self.new(self._records.values()[i])
        elif isinstance(i, int):
            return self._records.values()[i]
        else:
            raise RuntimeError(f"method __getitem__ for type {type(i)} not implemented")

    def __len__(self):
        return len(self._records)
