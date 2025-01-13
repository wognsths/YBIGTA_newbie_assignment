from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable

T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=list)
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        current_index = 0 
        for symbol in seq:
            found_child_index = None
            for child_idx in self[current_index].children:
                if self[child_idx].body == symbol:
                    found_child_index = child_idx
                    break

            if found_child_index is None:
                new_node_index = len(self)
                self.append(TrieNode(body=symbol))
                self[current_index].children.append(new_node_index)
                current_index = new_node_index
            else:

                current_index = found_child_index

        self[current_index].is_end = True