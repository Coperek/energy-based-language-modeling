from pathlib import Path
from typing import Generator, Literal

from itertools import batched
from unittest import case

import pyarrow.parquet as pq
import pyarrow as pa

import random


def shuffle(iterator) -> Generator:
    items = list(iterator)
    random.shuffle(items)

    for item in items:
        yield item

def build_batches(iterator, batch_size: int, drop_last: bool = False) -> Generator:
    out_tables = []

    for table in iterator:
        out_tables.append(table)

        if len(out_tables) >= batch_size:
            yield tuple(out_tables)
            out_tables = []

    if not drop_last:
        if len(out_tables) > 0:
            yield tuple(out_tables)


def disk_shuffle(iterator, shuffling_group_size: int, temp_path: str = "temp_shuffle.parquet"):

    batched_iterator = batched(iterator, shuffling_group_size)
    first_batch = next(batched_iterator)

    schema = first_batch[0].schema

    with pq.ParquetWriter(temp_path, schema, compression="snappy") as writer:

        shuffled_first_batch = list(first_batch)
        random.shuffle(shuffled_first_batch)
        writer.write_table(pa.concat_tables(shuffled_first_batch))

        for shuffling_group in batched_iterator:

            shuffled = list(shuffling_group)
            random.shuffle(shuffled)
            writer.write_table(pa.concat_tables(shuffled))


class ReaderConfig:
    def __init__(
        self,
        window_size: int = 1,
        window_overlap: int = 0,
        skip_first_n_items: int = 0,
        drop_last: bool = False,
        shuffle_mode: Literal["none", "disk", "ram"] = "none",
        disk_read_size: int = 65_536,
    ):
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.skip_first_n_items = skip_first_n_items
        self.drop_last = drop_last
        self.shuffle_mode = shuffle_mode
        self.disk_read_size = disk_read_size

        assert self.window_overlap < self.window_size
        assert self.skip_first_n_items >= 0
        assert self.shuffle_mode in ["none", "disk", "ram"]

    @property
    def is_shuffled(self):
        return self.shuffle_mode != "none"


class FileReader:
    _readers: dict[str, type["FileReader"]] = {}

    def __init__(self, path: str | Path, config: ReaderConfig = ReaderConfig()):
        self.path = path
        self.config = config

    @classmethod
    def register(cls, extension: str, reader_cls: type["FileReader"]):
        cls._readers[extension] = reader_cls

    def _read_records(self) -> Generator:
        raise NotImplementedError

    def _read(self) -> Generator:
        start_idx = self.config.skip_first_n_items
        items_left_to_read = self.config.window_size

        wip_table = None

        for batch in self._read_records():

            while start_idx < batch.num_rows:

                items_that_will_be_read = min(len(batch) - start_idx, items_left_to_read)
                read_table = batch.slice(start_idx, items_that_will_be_read)

                start_idx += items_that_will_be_read
                items_left_to_read -= items_that_will_be_read

                wip_table = read_table if wip_table is None else pa.concat_tables([wip_table, read_table])

                if items_left_to_read <= 0:
                    wip_table.combine_chunks()
                    yield wip_table

                    items_left_to_read = self.config.window_size - self.config.window_overlap
                    wip_table = wip_table.slice(wip_table.num_rows - self.config.window_overlap, self.config.window_overlap)

            start_idx -= batch.num_rows

        if not self.config.drop_last:
            if wip_table is not None and wip_table.num_rows > 0:
                wip_table.combine_chunks()
                yield wip_table

    def read(self) -> Generator:
        match self.config.shuffle_mode:
            case "disk":
                path_obj = Path(self.path)
                temp_path = path_obj.with_name(f"temp_shuffle_{path_obj.stem}.parquet")

                try:
                    disk_shuffle(self._read(), self.config.disk_read_size, temp_path=str(temp_path))
                    reader = self._readers[".parquet"](temp_path)

                    for item in reader.read():
                        yield item

                finally:
                    if temp_path.exists():
                        temp_path.unlink()

            case "ram":
                for item in shuffle(self._read()):
                    yield item

            case _:
                for item in self._read():
                    yield item

    def read_batched(self, batch_size: int, drop_last: bool = False) -> Generator:
        for table in build_batches(self.read(), batch_size, drop_last):
            yield table


class ParquetReader(FileReader):
    def __init__(
            self,
            path: str | Path,
            config: ReaderConfig = ReaderConfig(),
            columns: str | list = "all"
    ):
        super().__init__(path, config)

        self.pf = pq.ParquetFile(path, memory_map=True)
        self.meta = self.pf.metadata

        names = self.pf.schema_arrow.names
        if columns == "all":
            self.columns = names
        else:
            assert isinstance(columns, list)

            self.columns = list(columns)
            for i, val in enumerate(self.columns):
                if isinstance(val, int):
                    self.columns[i] = names[val]

    def __len__(self):
        return self.meta.num_rows

    def _read_records(self):
        if self.config.is_shuffled:
            row_group_indices = list(range(self.pf.num_row_groups))
            random.shuffle(row_group_indices)

            for rg_idx in row_group_indices:
                row_group_table = self.pf.read_row_group(rg_idx, columns=self.columns)

                batches = row_group_table.to_batches(max_chunksize=self.config.disk_read_size)

                for batch in batches:
                    yield pa.Table.from_batches([batch])
        else:
            for batch in self.pf.iter_batches(columns=self.columns, batch_size=self.config.disk_read_size):
                yield pa.Table.from_batches([batch])


FileReader.register(".parquet", ParquetReader)
