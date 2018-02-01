# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

"""Create scripts for experiments"""

import copy
import itertools
from argparse import ArgumentParser
import hashlib


def find_overlap_idx(list_of_strs, str_to_search):
    for (idx, cur_str) in enumerate(list_of_strs):
        if str_to_search in cur_str:
            return idx


def get_script_text(base_script, setting_dict, format_string):

    # Copy base script text
    script_text = copy.deepcopy(base_script)

    # Change hparam values
    for hparam_name in setting_dict.keys():
        idx = find_overlap_idx(script_text, hparam_name)
        script_text[idx] = format_string.format(hparam_name, setting_dict[hparam_name])

    # Remove trailing \ if present
    if script_text[-1].endswith('\\'):
        script_text[-1] = script_text[-1][:-1]

    return script_text


def get_short_name(field_name):
    return ''.join([a[0] for a in field_name.split('-')])


def get_filename(hparams, setting_dict, priority):
    filename_list = []
    for field, val in setting_dict.iteritems():
        short_name = get_short_name(field)
        filename_list.append(short_name + str(val))
    filename = '_'.join(filename_list)
    filename = hashlib.sha1(filename).hexdigest()
    filename = hparams.scripts_base_dir + priority + '_' + filename + '.sh'
    return filename


def write_script(filename, script_text):
    writer = open(filename, 'w')
    writer.write('\n'.join(script_text))
    writer.close()


def create_scripts(hparams, base_script, grid):
    num_scripts = 0
    priority = grid.pop('priority', '0')[0]
    for setting in itertools.product(*grid.values()):
        setting_dict = {}
        for (idx, value) in enumerate(setting):
            setting_dict[grid.keys()[idx]] = value
        format_string = hparams.format_string
        script_text = get_script_text(base_script, setting_dict, format_string)
        filename = get_filename(hparams, setting_dict, priority)
        write_script(filename, script_text)
        num_scripts += 1
    return num_scripts


def get_useful_lines(lines):
    # Remove blank lines and comments
    useful_lines = []
    for line in lines:
        ignore1 = (line[0] == '#')
        ignore2 = (line == '\n')
        ignore = ignore1 or ignore2
        if not ignore:
            useful_lines.append(line)
    return useful_lines


def parse_grid_spec(lines):
    """Parse the grid spec to get a list of grids."""

    useful_lines = get_useful_lines(lines)

    grids = []
    grid = {}
    for line in useful_lines:
        if line == '----\n':
            if len(grid) > 0:
                grids.append(grid)
            grid = {}
            continue
        line_split = line.split()
        field = line_split[0]
        values = line_split[1:]
        grid[field] = values

    # Append the final grid
    if len(grid) > 0:
        grids.append(grid)

    return grids


def main(hparams):
    print 'Creating scripts...'

    # Get the base script text
    with open(hparams.base_script_path, 'r') as f:
        base_script = [line[:-1] for line in f.readlines()]

    # Get grid spec text
    with open(hparams.grid_path, 'r') as f:
        lines = f.readlines()

    # Parse grid
    grids = parse_grid_spec(lines)
    print 'Number of grids = {}'.format(len(grids))

    # Create scripts
    for grid in grids:
        num_scripts = create_scripts(hparams, base_script, grid)
        print 'Number of scripts created = {}'.format(num_scripts)
    print 'Done\n'


if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--base-script-path', type=str, default='./base_script.sh', help='Path to base script')
    PARSER.add_argument('--grid-path', type=str, default='./grid.txt', help='Path to a file that specifies the grid')
    PARSER.add_argument('--format-string', type=str, default='{}={},\\', help='formatting specification')
    PARSER.add_argument('--scripts-base-dir', type=str, default='./scripts/',
                        help='Base directory to save scripts: Absolute path or relative to src')
    HPARAMS = PARSER.parse_args()
    main(HPARAMS)
