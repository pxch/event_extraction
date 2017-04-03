import logging


def get_console_logger(name, debug=False):
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Prepare a logger
    log = logging.getLogger(name)
    log.setLevel(level)

    if not log.handlers:
        # Just log to the console
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        # Put a timestamp on everything
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh.setFormatter(formatter)

        log.addHandler(sh)

    return log


def split_sections(input_iter, section_heads):
    """
    Divide up the lines of a file by searching for the given section heads (whole lines) in the
    order they're given.

    A list of the lines of each section is returned in a list, in the same order as the given section
    head list.

    If the given heads are not all found, None values are returned in place of those sections
    (which will be at the end of the list). The number of returned sections will always be
    len(section_heads)+1 -- an extra one for the text before the first head.

    Note that, although this is designed for use with lines of text, there's nothing about it specific
    to text: the objects in the list (and section head list) could be of any type.

    """
    input_iter = iter(input_iter)
    section_heads = iter(section_heads)
    next_head = section_heads.next()
    sections = [[]]

    try:
        for line in input_iter:
            if line == next_head:
                # Move onto the next section
                sections.append([])
                next_head = section_heads.next()
            else:
                # Add this line to the current section
                sections[-1].append(line)
    except StopIteration:
        # Reached the end of the list of section names: include the remainder of the input in the last section
        sections[-1].extend(list(input_iter))

    # Pad out the sections if there are heads we didn't use
    remaining_heads = list(input_iter)
    if remaining_heads:
        sections.extend([None] * len(remaining_heads))

    return sections
