# import argparse
# import os
# from datetime import datetime

# import gradio as gr
# from app import submit_function, person_example_fn
# # from app_flux import submit_function_flux


# HEADER = """
# <h1 style="text-align: center;"> Virtual-TryOn: Concatenation Is All You Need for Virtual Try-On with Diffusion Models </h1>
# """

# def app_gradio():
#     with gr.Blocks(title="Virtual-TryOn") as demo:
#         gr.Markdown(HEADER)
#         with gr.Row():
#             model_choice = gr.Dropdown(
#                 label="Select Model",
#                 choices=["sd", "flux"],
#                 value="sd",
#                 interactive=True
#             )

#         with gr.Row():
#             with gr.Column(scale=1, min_width=350):
#                 with gr.Row():
#                     image_path = gr.Image(type="filepath", interactive=True, visible=False)
#                     person_image = gr.ImageEditor(interactive=True, label="Person Image", type="filepath")

#                 with gr.Row():
#                     with gr.Column(scale=1, min_width=230):
#                         cloth_image = gr.Image(interactive=True, label="Condition Image", type="filepath")
#                     with gr.Column(scale=1, min_width=120):
#                         gr.Markdown(
#                             '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Try-On Cloth Type` to generate automatically </span>'
#                         )
#                         cloth_type = gr.Radio(
#                             label="Try-On Cloth Type",
#                             choices=["upper", "lower", "overall"],
#                             value="upper",
#                         )

#                 submit = gr.Button("Submit")
#                 gr.Markdown(
#                     '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
#                 )
                
#                 with gr.Accordion("Advanced Options", open=False):
#                     num_inference_steps = gr.Slider(
#                         label="Inference Step", minimum=10, maximum=100, step=5, value=50
#                     )
#                     guidance_scale = gr.Slider(
#                         label="CFG Strenth", minimum=0.0, maximum=50, step=0.5, value=2.5
#                     )
#                     seed = gr.Slider(
#                         label="Seed", minimum=-1, maximum=10000, step=1, value=42
#                     )
#                     # show_type = gr.Radio(
#                     #     label="Show Type",
#                     #     choices=["result only", "input & result", "input & mask & result"],
#                     #     value="result only",
#                     # )

#             with gr.Column(scale=2, min_width=500):
#                 with gr.Tabs():
#                     with gr.Tab("Masked Image"):
#                         masked_image = gr.Image(interactive=False, label="Masked Image")
#                     with gr.Tab("Result Image"):
#                         result_image = gr.Image(interactive=False, label="Result")
#                 with gr.Row():
#                     root_path = "resource/demo/example"
#                     with gr.Column():
#                         gr.Examples(
#                             examples=[os.path.join(root_path, "person", "men", f) for f in os.listdir(os.path.join(root_path, "person", "men"))],
#                             examples_per_page=4,
#                             inputs=image_path,
#                             label="Person Examples ①"
#                         )
#                         gr.Examples(
#                             examples=[os.path.join(root_path, "person", "women", f) for f in os.listdir(os.path.join(root_path, "person", "women"))],
#                             examples_per_page=4,
#                             inputs=image_path,
#                             label="Person Examples ②"
#                         )
#                         gr.Markdown(
#                             '<span style="color: #808080; font-size: small;">*Person examples from <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a>.</span>'
#                         )
#                     with gr.Column():
#                         gr.Examples(
#                             examples=[os.path.join(root_path, "condition", "upper", f) for f in os.listdir(os.path.join(root_path, "condition", "upper"))],
#                             examples_per_page=4,
#                             inputs=cloth_image,
#                             label="Condition Upper Examples"
#                         )
#                         gr.Examples(
#                             examples=[os.path.join(root_path, "condition", "overall", f) for f in os.listdir(os.path.join(root_path, "condition", "overall"))],
#                             examples_per_page=4,
#                             inputs=cloth_image,
#                             label="Condition Overall Examples"
#                         )
#                         gr.Examples(
#                             examples=[os.path.join(root_path, "condition", "person", f) for f in os.listdir(os.path.join(root_path, "condition", "person"))],
#                             examples_per_page=4,
#                             inputs=cloth_image,
#                             label="Condition Reference Person Examples"
#                         )

#         image_path.change(person_example_fn, inputs=image_path, outputs=person_image)

#         def submit_router(person, cloth, cloth_type_val, steps, scale, seed_val, show, model):
#             # if model == "sd":
#             result, mask = submit_function(person, cloth, cloth_type_val, steps, scale, seed_val, show)
#             return mask, result
#             # else:
#             #     from app_flux import submit_function_flux
#             #     result, mask = submit_function_flux(person, cloth, cloth_type_val, steps, scale, seed_val, show)

#         submit.click(
#             submit_router,
#             inputs=[
#                 person_image,
#                 cloth_image,
#                 cloth_type,
#                 num_inference_steps,
#                 guidance_scale,
#                 seed,
#                 # show_type,
#                 model_choice
#             ],
#             outputs=[masked_image, result_image]    
#         )

#     demo.queue().launch(share=True, show_error=True)



# if __name__ == "__main__":
#     app_gradio()

import argparse
import os
from datetime import datetime

import gradio as gr
from app import submit_function, person_example_fn
# from app_flux import submit_function_flux
# from app_pose import pose_transfer_function
# from app_style_pose import style_and_pose_transfer_function

HEADER = """
<h1 style="text-align: center;"> Virtual-TryOn: Concatenation Is All You Need for Virtual Try-On with Diffusion Models </h1>
"""

def app_gradio():
    with gr.Blocks(title="Virtual-TryOn") as demo:
        gr.Markdown(HEADER)

        with gr.Row():
            task_choice = gr.Dropdown(
                label="Select Task",
                choices=["style transfer", "pose transfer", "style and pose transfer"],
                value="style transfer",
                interactive=True
            )

            # model_choice = gr.Dropdown(
            #     label="Select Model",
            #     choices=["sd", "flux"],
            #     value="sd",
            #     interactive=True
            # )

        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    image_path = gr.Image(type="filepath", interactive=True, visible=False)
                    person_image = gr.Image(interactive=True, label="Person Image", type="filepath")

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(interactive=True, label="Condition Image", type="filepath")
                    with gr.Column(scale=1, min_width=120):
                        # gr.Markdown(
                        #     '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Try-On Cloth Type` to generate automatically </span>'
                        # )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )

                pose_image = gr.Image(interactive=True, label="Pose Image", type="filepath", visible=False)

                submit = gr.Button("Submit")
                gr.Markdown(
                    '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
                )

                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(
                        label="Inference Step", minimum=10, maximum=100, step=5, value=50
                    )
                    guidance_scale = gr.Slider(
                        label="CFG Strenth", minimum=0.0, maximum=50, step=0.5, value=2.5
                    )
                    seed = gr.Slider(
                        label="Seed", minimum=-1, maximum=10000, step=1, value=42
                    )
                    # show_type = gr.Radio(
                    #     label="Show Type",
                    #     choices=["result only", "input & result", "input & mask & result"],
                    #     value="result only",
                    # )

            with gr.Column(scale=2, min_width=500):
                with gr.Tabs():
                    with gr.Tab("Masked Image"):
                        mask_image = gr.Image(interactive=False, label="Masked Image")
                    with gr.Tab("Result Image"):
                        result_image = gr.Image(interactive=False, label="Result")

                with gr.Row():
                    root_path = "resource/demo/example"
                    with gr.Column():
                        gr.Examples(
                            examples=[os.path.join(root_path, "person", "men", f) for f in os.listdir(os.path.join(root_path, "person", "men"))],
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples ①"
                        )
                        gr.Examples(
                            examples=[os.path.join(root_path, "person", "women", f) for f in os.listdir(os.path.join(root_path, "person", "women"))],
                            examples_per_page=4,
                            inputs=image_path,
                            label="Person Examples ②"
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Person examples from <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a>.</span>'
                        )
                    with gr.Column():
                        gr.Examples(
                            examples=[os.path.join(root_path, "condition", "upper", f) for f in os.listdir(os.path.join(root_path, "condition", "upper"))],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Upper Examples"
                        )
                        gr.Examples(
                            examples=[os.path.join(root_path, "condition", "overall", f) for f in os.listdir(os.path.join(root_path, "condition", "overall"))],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Overall Examples"
                        )
                        gr.Examples(
                            examples=[os.path.join(root_path, "condition", "person", f) for f in os.listdir(os.path.join(root_path, "condition", "person"))],
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Reference Person Examples"
                        )

        image_path.change(person_example_fn, inputs=image_path, outputs=person_image)

        # Dynamically show/hide pose image
        def toggle_pose_input(task):
            return gr.update(visible=(task == "style and pose transfer"))
        task_choice.change(toggle_pose_input, inputs=task_choice, outputs=pose_image)

        # Route based on task and model
        def submit_router(person, cloth, cloth_type_val, steps, scale, seed_val, task, pose):
            if task == "style transfer":
                # if model == "sd":
                mask, result = submit_function(person, cloth, cloth_type_val, steps, scale, seed_val)
                return mask, result
            #     else:
            #         from app_flux import submit_function_flux
            #         return submit_function_flux(person, cloth, cloth_type_val, steps, scale, seed_val, show)

            # elif task == "pose transfer":
            #     from app_pose import pose_transfer_function
            #     return pose_transfer_function(person, cloth, cloth_type_val, steps, scale, seed_val, show)

            # elif task == "style and pose transfer":
            #     from app_style_pose import style_and_pose_transfer_function
            #     return style_and_pose_transfer_function(person, cloth, cloth_type_val, pose, steps, scale, seed_val, show)

        submit.click(
            submit_router,
            inputs=[
                person_image,
                cloth_image,
                cloth_type,
                num_inference_steps,
                guidance_scale,
                seed,
                task_choice,
                pose_image
            ],
            outputs=[mask_image, result_image],
        )

    demo.queue().launch(share=True, show_error=True)

if __name__ == "__main__":
    app_gradio()
