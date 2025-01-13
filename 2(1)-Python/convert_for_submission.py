PATH_1 = "./1-divide-and-conquer-multiplication"
PATH_2 = "./2-trie"
PATH_3 = "./3-segment-tree"

ROOT_PATH = {
    "10830": PATH_1,
    "3080": PATH_2,
    "5670": PATH_2,
    "2243": PATH_3,
    "3653": PATH_3,
    "17408": PATH_3
}

PATH_SUB = "./submission"


def f(n: str) -> None:
    with open(f"{ROOT_PATH[n]}/{n}.py", 'r', encoding='utf-8') as f_num:
        num_lines = f_num.readlines()

    num_code = "".join(filter(lambda x: "from lib import" not in x, num_lines))
    
    with open(f"{ROOT_PATH[n]}/lib.py", 'r', encoding='utf-8') as f_lib:
        lib_code = f_lib.read()
    
    code = lib_code + "\n\n\n" + num_code

    with open(f"{PATH_SUB}/{n}.py", 'w', encoding='utf-8') as f_out:
        f_out.write(code)


if __name__ == "__main__":
    for k in ROOT_PATH:
        f(k)
