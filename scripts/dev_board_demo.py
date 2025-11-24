from cosee.board import Board, View


def main() -> None:
    board = Board()

    board.add_cell(
        view=View(page=1),
        content="Document title: CoSee Whiteboard",
        tags=["title"],
        author="Scanner",
        step=0,
    )
    board.add_cell(
        view=View(page=1, description="upper table"),
        content="Table 1: accuracy numbers for baseline vs CoSee.",
        tags=["table"],
        author="DetailReader",
        step=1,
    )
    board.add_cell(
        view=View(page=2, description="intro paragraph"),
        content="Intro mentions multimodal collaboration with shared board.",
        tags=["paragraph"],
        author="Scanner",
        step=1,
    )
    board.add_cell(
        view=View(page=3, description="conclusion section"),
        content="Verified final answer against table and text.",
        tags=["check"],
        author="CrossChecker",
        step=2,
    )

    summary = board.to_text(max_cells_per_page=8, max_total_chars=500)
    print("BOARD SUMMARY:")
    print(summary)


if __name__ == "__main__":
    main()
