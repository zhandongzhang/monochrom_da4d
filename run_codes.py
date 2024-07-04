import sys
from pathlib import Path
import json
import subprocess
from generic_parser import EntryPointParameters, entrypoint
import jinja2

REPOSITORY_THIS_LEVEL = Path(__file__).resolve().parent
REFERENCE_FILE = json.load(open(REPOSITORY_THIS_LEVEL/"reference_parameters.json", encoding="utf-8"))
TEMPLATE_DIRECTORY = REPOSITORY_THIS_LEVEL/'templates'

MADX_SCRIPTS = {
    'executable': REPOSITORY_THIS_LEVEL/'codes'/'madx',
    'templates': ['madx_thin.template'],
    }

PYTHON_SCRIPTS = {
    'executable': sys.executable,
    'templates': [  
                    'xsuite_translation.template',
                    'xsuite_tracking.template',
                    'xsuite_dynap_xy.template',
                    'xsuite_plotting_xy.template',
                    'xsuite_dynap_xz.template',
                    'xsuite_plotting_xz.template',
                  ],
    }

SCRIPTS = [MADX_SCRIPTS, PYTHON_SCRIPTS]


# Script arguments -------------------------------------------------------------
def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="operation_mode",
        type=str,
        required=True,
        choices=['zmh4ip','zmv1','zmm'],
        help="Define which operation mode should be tested: zmh4ip,zmv1,zmm",
    )

    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    working_directory = REPOSITORY_THIS_LEVEL/"public"/opt.operation_mode
    working_directory.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        for template in script['templates']:
            fill_template(template, opt.operation_mode, working_directory)
            print(f'Running {script["executable"]} {working_directory}/{template} ------------------')
            run_script(script['executable'], template, working_directory)


def fill_template(template_file, operation_mode, working_directory):
    loader = jinja2.FileSystemLoader(searchpath=TEMPLATE_DIRECTORY)
    env = jinja2.Environment(loader=loader, undefined=jinja2.StrictUndefined)
    template = env.get_template(template_file)
    with open(working_directory/return_filled_mask_name(template_file), 'w') as f:
        f.write(template.render(reference=REFERENCE_FILE[operation_mode], operation_mode=operation_mode))


def run_script(executable, template_file, working_directory):
    with open(working_directory/return_log_name(template_file), 'w') as f:
        subprocess.run([executable, return_filled_mask_name(template_file)], check=True, stdout=f, cwd=working_directory)


def return_filled_mask_name(template_file):
    return Path(template_file).with_suffix('').with_suffix('.rendered')


def return_log_name(template_file):
    return Path(template_file).with_suffix('').with_suffix('.log')


if __name__ == "__main__":
    main()
