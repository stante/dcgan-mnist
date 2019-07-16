import torch
import model as m
import click


@click.command()
@click.argument('input-file')
@click.argument('output-file')
@click.option('--verbose', default=True)
def main(input_file, output_file, verbose):
    loaded_model = torch.load(input_file)
    model = m.LinearModelGenerator(loaded_model['latent_dim'])
    model.load_state_dict(loaded_model['state_dict'])
    model.eval()

    z = torch.rand((1, 100)) * 2 - 1

    torch.onnx.export(model, z, output_file, verbose=verbose)


if __name__ == '__main__':
    main()
