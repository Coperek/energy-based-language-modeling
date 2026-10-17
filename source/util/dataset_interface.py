from typing import Generator, Literal, Iterable, Union

from pathlib import Path
from itertools import batched, accumulate

import uuid
import math
import random

import pyarrow.parquet as pq
import pyarrow as pa


def pyarrow_concat(
        inputs: Iterable[Union[pa.Table, pa.Array, pa.ChunkedArray]],
        zero_copy_arrays: bool = True
) -> Union[pa.Table, pa.Array, pa.ChunkedArray]:
    """
    Unified function to concatenate either Tables, Arrays, or ChunkedArrays.
    Requires all inputs within the collection to share the same underlying type category.
    """
    inputs_list = list(inputs)
    if not inputs_list:
        raise ValueError("The input sequence cannot be empty.")

    first = inputs_list[0]

    if isinstance(first, pa.Table):
        return pa.concat_tables(inputs_list)
    elif isinstance(first, (pa.Array, pa.ChunkedArray)):
        if zero_copy_arrays:
            return pa.chunked_array(inputs_list)
        return pa.concat_arrays(inputs_list)
    else:
        raise TypeError(f"Unsupported PyArrow type for unification: {type(first)}")


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


def windows_to_table(windows) -> pa.Table:
    """List of window tables -> table with one row per window (list-typed columns)."""
    big = pyarrow_concat(windows, zero_copy_arrays=False)
    offsets = pa.array([0, *accumulate(w.num_rows for w in windows)], type=pa.int32())

    arrays = [
        pa.ListArray.from_arrays(offsets, big.column(name).combine_chunks())
        for name in big.column_names
    ]
    return pa.Table.from_arrays(arrays, names=big.column_names)


def table_to_window(row_table: pa.Table) -> pa.Table:
    """1-row table with list columns -> window table."""
    return pa.table({n: row_table.column(n)[0].values for n in row_table.column_names})


def disk_shuffle(iterator, shuffling_group_size: int, temp_path: str):
    batched_iterator = batched(iterator, shuffling_group_size)
    first_group = next(batched_iterator, None)
    if first_group is None:
        return

    first_group = list(first_group)
    random.shuffle(first_group)
    first_table = windows_to_table(first_group)

    with pq.ParquetWriter(temp_path, first_table.schema, compression="snappy") as writer:
        writer.write_table(first_table)

        for group in batched_iterator:
            group = list(group)
            random.shuffle(group)
            writer.write_table(windows_to_table(group))


def validate_path(path: Path, desired_suffix: str):
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix != desired_suffix and not path.is_dir():
        raise ValueError(f"Error: wrong file extension, expected: {desired_suffix}, got: {path.suffix}")

    return True


class ReaderConfig:
    def __init__(
        self,
        window_size: int | None = None,
        window_overlap: int = 0,
        skip_first_n_items: int = 0,
        drop_last: bool = False,
        shuffle_mode: Literal["none", "disk", "ram"] = "none",
        output_mode: Literal["pyarrow", "python", "pandas"] = "pyarrow",
        disk_read_size: int = 4096,
        logs: bool = False,
    ):
        self._window_size = window_size
        self.window_overlap = window_overlap
        self.skip_first_n_items = skip_first_n_items
        self.drop_last = drop_last
        self.shuffle_mode = shuffle_mode
        self.output_mode = output_mode
        self.disk_read_size = disk_read_size
        self.logs = logs

        assert self.window_overlap < self.window_size
        assert self.skip_first_n_items >= 0
        assert self.shuffle_mode in ["none", "disk", "ram"]
        assert self.output_mode in ["pyarrow", "python", "pandas"]
        
    @property
    def window_size(self):
        return 1 if self._window_size is None else self._window_size

    @property
    def is_shuffled(self):
        return self.shuffle_mode != "none"

    @property
    def step(self):
        return self.window_size - self.window_overlap

    def calculate_window_count(self, item_count):
        n = max(item_count - self.skip_first_n_items, 0)
        w, s = self.window_size, self.step

        if n < w:
            return 0 if (self.drop_last or n == 0) else 1

        full = (n - w) // s + 1
        if self.drop_last:
            return full
        return full + (1 if (full - 1) * s + w < n else 0)

    def convert_output(self, iterator):
        match self.output_mode:
            case "python":
                for _ in iterator:
                    yield _.to_pylist()[0] if self.window_size == 1 else _.to_pylist()
            case "pandas":
                for _ in iterator:
                    yield _.to_pandas().iloc(0) if self.window_size == 1 else _.to_pandas()
            case _:
                for _ in iterator:
                    yield _ # TODO


class Reader:
    def __init__(self):
        return

    def _read_records(self) -> Generator:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def read(self) -> Generator:
        raise NotImplementedError

    def read_batched(self, batch_size: int, drop_last: bool = False) -> Generator:
        for table in build_batches(self.read(), batch_size, drop_last):
            yield table


class CompositionReader(Reader):
    def __init__(self, readers_and_shares: dict[Reader, float], mode: Literal["shrink", "expand"] = "shrink"):
        total = sum(readers_and_shares.values())
        self.readers_and_shares = {r: s / total for r, s in readers_and_shares.items()}

        assert mode in ["shrink", "expand"]
        self.mode = mode

        super().__init__()

    def _plan(self):
        """Returns (readers, quotas); quotas[i] lists how many items reader i yields in each epoch."""
        items = [(r, s, len(r)) for r, s in self.readers_and_shares.items()]
        items = [(r, s, n) for r, s, n in items if n > 0]  # skip empty readers
        if not items:
            return [], []

        readers = [r for r, _, _ in items]
        ratios = [n / s for _, s, n in items]
        total = min(ratios) if self.mode == "shrink" else max(ratios)

        quotas = []
        for _, s, n in items:
            epochs = s * total / n
            q = []
            while epochs > 1e-9:
                q.append(math.floor(n * min(1.0, epochs) + 1e-6))
                epochs -= 1
            quotas.append(q)
        return readers, quotas

    def __len__(self):
        _, quotas = self._plan()
        return sum(sum(q) for q in quotas)

    def read(self) -> Generator:
        readers, quotas = self._plan()

        epoch = [0] * len(readers)
        gens, left = {}, {}

        def start_epoch(i):
            gens[i] = readers[i].read()
            left[i] = quotas[i][epoch[i]]
            epoch[i] += 1

        active = [i for i in range(len(readers)) if quotas[i]]
        for i in active:
            start_epoch(i)

        while True:
            for i in list(active):
                while left[i] <= 0:
                    gens[i].close()
                    if epoch[i] < len(quotas[i]):
                        start_epoch(i)
                    else:
                        active.remove(i)
                        break
            if not active:
                return

            i = random.choices(active, weights=[left[j] for j in active])[0]
            try:
                item = next(gens[i])
            except StopIteration:
                left[i] = 0
                continue
            left[i] -= 1
            yield item


class WeightedReader(CompositionReader):
    def __init__(self, readers_and_weights: dict[Reader, float], mode: Literal["shrink", "expand"] = "shrink"):
        total = sum([len(r) * w for r, w in readers_and_weights.items()])
        super().__init__({r: len(r) * w / total for r, w in readers_and_weights.items()}, mode)


class GroupReader(WeightedReader):
    def __init__(self, readers: list[Reader]):
        super().__init__({r: 1.0 for r in readers})


class FileReader(Reader):
    _readers: dict[str, type["FileReader"]] = {}

    @classmethod
    def register(cls, extension: str, reader_cls: type["FileReader"]):
        cls._readers[extension] = reader_cls

    def __new__(cls, path: str, config: ReaderConfig = ReaderConfig(), *args, **kwargs):
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            raise FileNotFoundError(f"ERROR: Nothing found at path {path}")

        if path_obj.is_dir():
            return GroupReader([FileReader(str(p), config) for p in path_obj.iterdir()])

        extension = path_obj.suffix
        if extension not in cls._readers:
            raise ValueError(f"ERROR: Unsupported extension: {extension}")

        if config.logs:
            print("Registered file \"" + str(path) + "\"")

        return super().__new__(cls._readers[extension])

    def __init__(self, path: str, config: ReaderConfig = ReaderConfig()):
        super().__init__()
        self.config = config
        self.path = path

    def _read(self) -> Generator:
        start_idx = self.config.skip_first_n_items
        items_left_to_read = self.config.window_size

        fresh = 0
        wip_table = None

        for batch in self._read_records():

            while start_idx < len(batch):

                items_that_will_be_read = min(len(batch) - start_idx, items_left_to_read)
                read_table = batch.slice(start_idx, items_that_will_be_read)

                fresh += items_that_will_be_read
                start_idx += items_that_will_be_read
                items_left_to_read -= items_that_will_be_read

                wip_table = read_table if wip_table is None else pyarrow_concat([wip_table, read_table], zero_copy_arrays=False)

                if items_left_to_read <= 0:

                    yield wip_table
                    fresh = 0

                    items_left_to_read = self.config.window_size - self.config.window_overlap
                    wip_table = wip_table.slice(len(wip_table) - self.config.window_overlap,
                                                self.config.window_overlap)

            start_idx -= len(batch)

        if not self.config.drop_last and fresh > 0:
            if wip_table is not None and wip_table.num_rows > 0:
                yield wip_table

    def read(self) -> Generator:
        match self.config.shuffle_mode:
            case "disk":
                temp_path = f"temp_shuffle_{Path(self.path).stem}{uuid.uuid4().hex[:8]}.parquet"

                try:
                    disk_shuffle(self._read(), self.config.disk_read_size, temp_path=temp_path)
                    temp_reader = self._readers[".parquet"](temp_path)

                    for row in temp_reader.read():
                        yield from self.config.convert_output([table_to_window(row)])

                finally:
                    temp_path_obj = Path(temp_path)
                    if temp_path_obj.exists():
                        temp_path_obj.unlink()

            case "ram":
                for row in self.config.convert_output(shuffle(self._read())):
                    yield row

            case _:
                for row in self.config.convert_output(self._read()):
                    yield row


class ParquetReader(FileReader):
    def __new__(
            cls,
            path: str,
            config: ReaderConfig = ReaderConfig(),
            columns: None | str | int | list = None
    ):
        path_obj = Path(path).resolve()
        validate_path(path_obj, ".parquet")

        if path_obj.is_dir():
            return GroupReader([ParquetReader(str(p), config, columns) for p in path_obj.iterdir()])

        return super().__new__(ParquetReader, path=path, config=config)

    def __init__(
            self,
            path: str,
            config: ReaderConfig = ReaderConfig(),
            columns: None | str | int | list = None
    ):
        super().__init__(path, config)

        self.pf = pq.ParquetFile(path, memory_map=True)
        self.meta = self.pf.metadata

        self.output_single_column = False
        names = self.pf.schema_arrow.names
        if columns is None:
            self.columns = names
        elif isinstance(columns, str):
            self.columns = [columns]
            self.output_single_column = True
        elif isinstance(columns, int):
            self.columns = [names[columns]]
            self.output_single_column = True
        else:
            assert isinstance(columns, list)

            self.columns = list(columns)
            for i, val in enumerate(self.columns):
                if isinstance(val, int):
                    self.columns[i] = names[val]

    def __len__(self):
        return self.config.calculate_window_count(self.meta.num_rows)

    def _read_records(self):
        for batch in self.pf.iter_batches(columns=self.columns, batch_size=self.config.disk_read_size):
            if self.output_single_column:
                yield batch.column(self.columns[0])
            else:
                yield pa.Table.from_batches([batch])


FileReader.register(".parquet", ParquetReader)
