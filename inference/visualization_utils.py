from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import os

import pymol2

import ipdb

def draw_mol_with_attn(mol, attn_scores, mol_id, attn_id, curr_out_folder):
    d_canv = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
    d_canv.drawOptions().addAtomIndices = True
    d_canv.drawOptions().highlightRadius = 0.5

    d_canv.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), 
                    highlightAtomColors={i: (0, 0, 1, float(score)) for i, score in enumerate(attn_scores)},
                    highlightBonds=[], highlightBondColors={})

    d_canv.FinishDrawing()
    mol_img = d_canv.GetDrawingText()
    mol_img_file = os.path.join(curr_out_folder, f"mol_{attn_id}_{mol_id}.png")
    with open(mol_img_file, 'wb') as f:
        f.write(mol_img)


def draw_protein_with_attn(prot_pdb, attn_scores, prot_id, attn_id,
                           min_attn=None,
                           max_attn=None,
                           pymol_context=None,
                           close_context=True,
                           curr_out_folder=None):
    
    if pymol_context is None:
        pymol_context = _init_pymol_context()

    if min_attn is None:
        min_attn = 0

    if max_attn is None:
        max_attn = max(attn_scores)

    legal_id = pymol_context.cmd.get_legal_name(prot_id)

    pymol_context.cmd.load(prot_pdb, legal_id)
    pymol_context.cmd.alter(legal_id, "b = 0")

    # Renumber all the residues to start at 1
    # and remove all the other molecules entirely
    pymol_context.cmd.remove("not polymer")

    # Set the B-factors for the protein
    for i, score in enumerate(attn_scores):
        pymol_context.cmd.alter(f"{legal_id} and resi {i+1}", f"b = {score}")

    # Set the color scheme for the protein to be based on the B-factors
    # with a minimum of 0 and a maximum of whatever the largest value was
    pymol_context.cmd.spectrum("b", "red yellow blue", f"{legal_id}", minimum=min_attn, maximum=max_attn)

    # TODO items:
    # make parts of protein with low attention scores more transparent 
    # add a color bar to the side of the protein/image somehow

    # Save an image of the protein if a folder is provided
    if curr_out_folder is not None:
        pymol_context.cmd.orient(legal_id)
        pymol_context.cmd.center(legal_id)
        pymol_context.cmd.zoom(legal_id, complete=1, buffer=2.0)

        pymol_context.cmd.ray(1800, 1200)
        
        ret_file = os.path.join(curr_out_folder, f"prot_{attn_id}_{prot_id}.png")
        pymol_context.cmd.png(ret_file, dpi=600)

    if close_context:
        pymol_context.stop()
        pymol_context = None

    return pymol_context


def draw_protein_difference(ref_pdb, alt_pdb, ref_attn, alt_attn,
                            base_id, curr_out_folder,
                            scale_attn_by_length=True,
                            logscale_attn=True,
                            n_top_labels=10):
    
    # Make a base Pymol context where all of the proteins will be loaded
    pymol_context = _init_pymol_context()

    # Define the IDs for the proteins
    # noting that we want legal names downstream
    base_legal_id = pymol_context.cmd.get_legal_name(base_id)

    ref_id = f"ref_{base_legal_id}"
    alt_id = f"alt_{base_legal_id}"
    diff_id = f"diff_{base_legal_id}"
    top_id = f"top_{base_legal_id}"

    # Make the attention vectors into numpy arrays
    ref_attn = np.array(ref_attn)
    alt_attn = np.array(alt_attn)

    ref_len = len(ref_attn)
    alt_len = len(alt_attn)

    # If we're scaling the attention scores to the length of the proteins
    # then we want to divide by the "base" attention score, which is 1/length (aka multiply by length)
    # (which basically gives the proportion of this attention over the expected uniform attention)
    if scale_attn_by_length:
        ref_attn = ref_attn * ref_len
        alt_attn = alt_attn * alt_len

    # Compute the ratio between the reference and alternate attentions, padding where appropriate
    # Pad the missing ones for alt with ones, and the missing ones for ref with ones (to avoid zero-division)
    # TODO: how do we pad for reference? Ones may not be appropriate)
    # TODO: we may also want to align the sequences to prevent padding at the wrong place? But for now assume unneeded
    longer_prot_len = max(ref_len, alt_len)
    ref_attn_pad = np.pad(ref_attn, (0, longer_prot_len - ref_len), mode='constant', constant_values=1)
    alt_attn_pad = np.pad(alt_attn, (0, longer_prot_len - alt_len), mode='constant', constant_values=1)

    diff_attn = alt_attn_pad / ref_attn_pad

    # Log-scale the attention ratios for visualization
    if logscale_attn:
        diff_attn = np.log10(diff_attn)
        ref_attn = np.log10(ref_attn)
        alt_attn = np.log10(alt_attn)

     # Get the maximum attention score between reference and alternate proteins
    max_attn = max(np.abs(ref_attn).max(), np.abs(alt_attn).max())
    max_diff_attn = max(np.abs(diff_attn))

    # Minimum attention is the negative of the maximum attention if log-scaled
    # Otherwise, is zero
    min_attn = -max_attn if logscale_attn else 0
    min_diff_attn = -max_diff_attn if logscale_attn else 0

    
    # Load each of the proteins into this context
    # (Do not save the images, as we'll save them all at once at the end)
    draw_protein_with_attn(ref_pdb, ref_attn, ref_id, base_id, 
                           min_attn=min_attn, max_attn=max_attn,
                           pymol_context=pymol_context, close_context=False, curr_out_folder=None)
    draw_protein_with_attn(alt_pdb, alt_attn, alt_id, base_id,
                           min_attn=min_attn, max_attn=max_attn,
                           pymol_context=pymol_context, close_context=False, curr_out_folder=None)
    draw_protein_with_attn(ref_pdb, diff_attn, diff_id, base_id,
                           min_attn=min_diff_attn, max_attn=max_diff_attn,
                           pymol_context=pymol_context, close_context=False, curr_out_folder=None)

    # Align the alternate protein to the reference protein
    pymol_context.cmd.align(alt_id, ref_id)

    # Recreate in grid form
    pymol_context.cmd.set("grid_mode", 1)

    # Orient, center, and zoom the camera once more
    pymol_context.cmd.orient(ref_id)
    pymol_context.cmd.center(ref_id)
    pymol_context.cmd.zoom(ref_id, complete=1, buffer=2.0)


    # Add residue number for the top highest attention ratios between ref and alt
    # We use the absolute log scale of the ratios to get the top 
    # using the cmd.label() function (after copying the diff protein to a new object)
    # Note argsort is 0-indexed, so we add 1 to get the 1-indexed residue number
    # (Note, we don't currently use the actual values beyond sorting 
    # because PyMol only supports labels of a single size)
    # TODO: color the labels by whether they're above the median or below?

    pymol_context.cmd.copy(top_id, diff_id)
    diff_attn_scales = np.abs(diff_attn)
    top_diffs = np.argsort(diff_attn_scales)[-n_top_labels:] + 1

    for i in top_diffs:
        pymol_context.cmd.label(f"{top_id} and resi {i} and name CA", f"{i}")

    # Show only the labels
    pymol_context.cmd.hide(selection=top_id)
    pymol_context.cmd.show("labels", top_id)

    # Run ray to get a better image
    pymol_context.cmd.ray(1800, 1200)

    ret_file = os.path.join(curr_out_folder, f"grid_{base_id}.png")

    pymol_context.cmd.png(ret_file, dpi=600)
    pymol_context.cmd.set("grid_mode", 0)
    
    # Kill this PyMol context after saving the images
    pymol_context.stop()


def _init_pymol_context():
    pymol_context = pymol2.PyMOL()
    pymol_context.start()

    cmd = pymol_context.cmd

    # Start pymol
    cmd.feedback("disable","all","actions")
    cmd.feedback("disable","all","results")
    cmd.feedback("disable","all","warnings")

    # PNG save arguments
    cmd.viewport(1800, 1200)
    cmd.set("ray_opaque_background", "off")
    cmd.set("ray_trace_mode", 0)
    cmd.set("ray_trace_gain", 0.0)
    cmd.set("orthoscopic", "off")
    # cmd.set("ray_trace_color", "black")
    cmd.set("ray_trace_fog", 0)
    cmd.set("ray_label_specular", 0)
    cmd.set("ray_shadows", 0)
    cmd.set("depth_cue", 0)
    cmd.set("antialias", 4)
    cmd.set("hash_max", 300)
    cmd.set("float_labels", 1)
    cmd.set("label_anchor", "CA")
    cmd.set("label_size", 30)
    # cmd.set("label_bg_color", "black")
    # cmd.set("label_bg_transparency", "0.0")
    cmd.set("label_color", 'white')

    return pymol_context