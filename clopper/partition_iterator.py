from typing import Iterator, Tuple, List


def partition_iterator(num_partitions: int, float_tuple: Tuple[float, ...]) -> Iterator[Tuple[Tuple[float, ...], ...]]:
    if num_partitions <= 0:
        return iter([])  # Empty iterator for invalid partition count

    def partitions(seq: Tuple[float, ...], k: int) -> Iterator[List[List[float]]]:
        if k == 1:
            yield [list(seq)]
        else:
            for i in range(1, len(seq)):
                for part in partitions(seq[i:], k - 1):
                    yield [list(seq[:i])] + part

    unique_partitions = set()

    for partition in partitions(float_tuple, num_partitions):
        partition_tuple: Tuple[Tuple[float, ...], ...] = tuple(tuple(sublist) for sublist in partition)
        if partition_tuple not in unique_partitions:
            unique_partitions.add(partition_tuple)
            yield partition_tuple


if __name__ == "__main__":
    float_tuple_ = (1.0, 2.0, 2.0, 3.0)
    num_partitions_ = 2
    partitions_ = partition_iterator(num_partitions_, float_tuple_)
    for p in partitions_:
        print(p)

    float_tuple_ = (1.0,)
    num_partitions_ = 3
    partitions_ = partition_iterator(num_partitions_, float_tuple_)
    print("Partitions:", tuple(partitions_))
