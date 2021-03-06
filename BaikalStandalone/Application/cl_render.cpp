/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#include "OpenImageIO/imageio.h"

#include "Application/cl_render.h"
#include "Application/gl_render.h"

#include "SceneGraph/scene1.h"
#include "SceneGraph/camera.h"
#include "SceneGraph/material.h"
#include "scene_io.h"
#include "material_io.h"
#include "SceneGraph/material.h"

#include "Renderers/monte_carlo_renderer.h"
#include "Renderers/adaptive_renderer.h"

#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>

#ifdef ENABLE_DENOISER
#include "PostEffects/wavelet_denoiser.h"
#endif
#include "Utils/clw_class.h"

namespace Baikal
{
    AppClRender::AppClRender(AppSettings& settings, GLuint tex) : m_tex(tex), m_output_type(Renderer::OutputType::kColor)
    {
        InitCl(settings, m_tex);
        LoadScene(settings);
    }

    void AppClRender::InitCl(AppSettings& settings, GLuint tex)
    {
        bool force_disable_itnerop = false;
        //create cl context
        try
        {
            ConfigManager::CreateConfigs(
                settings.mode,
                settings.interop,
                m_cfgs,
                settings.num_bounces,
                settings.platform_index,
                settings.device_index);
        }
        catch (CLWException &)
        {
            force_disable_itnerop = true;
            ConfigManager::CreateConfigs(settings.mode, false, m_cfgs, settings.num_bounces);
        }

        m_width = (std::uint32_t)settings.width;
        m_height = (std::uint32_t)settings.height;

        std::cout << "Running on devices: \n";

        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            std::cout << i << ": " << m_cfgs[i].context.GetDevice(0).GetName() << "\n";
        }

        settings.interop = false;

        m_outputs.resize(m_cfgs.size());
        m_ctrl.reset(new ControlData[m_cfgs.size()]);

        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            if (m_cfgs[i].type == ConfigManager::kPrimary)
            {
                m_primary = i;

                if (m_cfgs[i].caninterop)
                {
                    m_cl_interop_image = m_cfgs[i].context.CreateImage2DFromGLTexture(tex);
                    settings.interop = true;
                }
            }

            m_ctrl[i].clear.store(1);
            m_ctrl[i].stop.store(0);
            m_ctrl[i].newdata.store(0);
            m_ctrl[i].idx = i;
        }

        if (force_disable_itnerop)
        {
            std::cout << "OpenGL interop is not supported, disabled, -interop flag is ignored\n";
        }
        else
        {
            if (settings.interop)
            {
                std::cout << "OpenGL interop mode enabled\n";
            }
            else
            {
                std::cout << "OpenGL interop mode disabled\n";
            }
        }

        //create renderer
#pragma omp parallel for
        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            m_outputs[i].output = m_cfgs[i].factory->CreateOutput(m_width, m_height);

			m_outputs[i].output_position = m_cfgs[i].factory->CreateOutput(settings.width, settings.height);
			m_outputs[i].output_normal = m_cfgs[i].factory->CreateOutput(settings.width, settings.height);
			m_outputs[i].output_albedo = m_cfgs[i].factory->CreateOutput(settings.width, settings.height);

#ifdef ENABLE_DENOISER
            m_outputs[i].output_denoised = m_cfgs[i].factory->CreateOutput(settings.width, settings.height);
            m_outputs[i].output_normal = m_cfgs[i].factory->CreateOutput(settings.width, settings.height);
            m_outputs[i].output_position = m_cfgs[i].factory->CreateOutput(settings.width, settings.height);
            m_outputs[i].output_albedo = m_cfgs[i].factory->CreateOutput(settings.width, settings.height);
            m_outputs[i].output_mesh_id = m_cfgs[i].factory->CreateOutput(settings.width, settings.height);

            //m_outputs[i].denoiser = m_cfgs[i].factory->CreatePostEffect(Baikal::RenderFactory<Baikal::ClwScene>::PostEffectType::kBilateralDenoiser);
            m_outputs[i].denoiser = m_cfgs[i].factory->CreatePostEffect(Baikal::RenderFactory<Baikal::ClwScene>::PostEffectType::kWaveletDenoiser);
#endif
            m_cfgs[i].renderer->SetOutput(Baikal::Renderer::OutputType::kColor, m_outputs[i].output.get());

			m_cfgs[i].renderer->SetOutput(Baikal::Renderer::OutputType::kWorldPosition, m_outputs[i].output_position.get());
			m_cfgs[i].renderer->SetOutput(Baikal::Renderer::OutputType::kWorldShadingNormal, m_outputs[i].output_normal.get());
			m_cfgs[i].renderer->SetOutput(Baikal::Renderer::OutputType::kAlbedo, m_outputs[i].output_albedo.get());

#ifdef ENABLE_DENOISER
            m_cfgs[i].renderer->SetOutput(Baikal::Renderer::OutputType::kWorldShadingNormal, m_outputs[i].output_normal.get());
            m_cfgs[i].renderer->SetOutput(Baikal::Renderer::OutputType::kWorldPosition, m_outputs[i].output_position.get());
            m_cfgs[i].renderer->SetOutput(Baikal::Renderer::OutputType::kAlbedo, m_outputs[i].output_albedo.get());
            m_cfgs[i].renderer->SetOutput(Baikal::Renderer::OutputType::kMeshID, m_outputs[i].output_mesh_id.get());
#endif

            m_outputs[i].fdata.resize(settings.width * settings.height);
            m_outputs[i].udata.resize(settings.width * settings.height * 4);

            if (m_cfgs[i].type == ConfigManager::kPrimary)
            {
                m_outputs[i].copybuffer = m_cfgs[i].context.CreateBuffer<RadeonRays::float3>(m_width * m_height, CL_MEM_READ_WRITE);				
            }
        }

        m_shape_id_data.output = m_cfgs[m_primary].factory->CreateOutput(m_width, m_height);
        m_cfgs[m_primary].renderer->Clear(RadeonRays::float3(0, 0, 0), *m_outputs[m_primary].output);
        m_cfgs[m_primary].renderer->Clear(RadeonRays::float3(0, 0, 0), *m_shape_id_data.output);		
    }

	const char *getErrorString(cl_int error)
	{
		switch (error) {
			// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

			// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

			// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
		}
	}

    void AppClRender::LoadScene(AppSettings& settings)
    {
        rand_init();

        // Load obj file
        std::string basepath = settings.path;
        basepath += "/";
        std::string filename = basepath + settings.modelname;

        {
            m_scene = Baikal::SceneIo::LoadScene(filename, basepath);
            // Enable this to generate new materal mapping for a model
#if 0
            auto material_io{Baikal::MaterialIo::CreateMaterialIoXML()};
            material_io->SaveMaterialsFromScene(basepath + "materials.xml", *m_scene);
            material_io->SaveIdentityMapping(basepath + "mapping.xml", *m_scene);
#endif

            // Check it we have material remapping
            std::ifstream in_materials(basepath + "materials.xml");
            std::ifstream in_mapping(basepath + "mapping.xml");

            if (in_materials && in_mapping)
            {
                in_materials.close();
                in_mapping.close();

                auto material_io = Baikal::MaterialIo::CreateMaterialIoXML();
                auto mats = material_io->LoadMaterials(basepath + "materials.xml");
                auto mapping = material_io->LoadMaterialMapping(basepath + "mapping.xml");

                material_io->ReplaceSceneMaterials(*m_scene, *mats, mapping);
            }
        }

        switch (settings.camera_type)
        {
        case CameraType::kPerspective:
            m_camera = Baikal::PerspectiveCamera::Create(
                settings.camera_pos
                , settings.camera_at
                , settings.camera_up);

            break;
        case CameraType::kOrthographic:
            m_camera = Baikal::OrthographicCamera::Create(
                settings.camera_pos
                , settings.camera_at
                , settings.camera_up);
            break;
        default:
            throw std::runtime_error("AppClRender::InitCl(...): unsupported camera type");
        }

        m_scene->SetCamera(m_camera);

        // Adjust sensor size based on current aspect ratio
        float aspect = (float)settings.width / settings.height;
        settings.camera_sensor_size.y = settings.camera_sensor_size.x / aspect;

        m_camera->SetSensorSize(settings.camera_sensor_size);
        m_camera->SetDepthRange(settings.camera_zcap);

        auto perspective_camera = std::dynamic_pointer_cast<Baikal::PerspectiveCamera>(m_camera);

        // if camera mode is kPerspective
        if (perspective_camera)
        {
            perspective_camera->SetFocalLength(settings.camera_focal_length);
            perspective_camera->SetFocusDistance(settings.camera_focus_distance);
            perspective_camera->SetAperture(settings.camera_aperture);
            std::cout << "Camera type: " << (perspective_camera->GetAperture() > 0.f ? "Physical" : "Pinhole") << "\n";
            std::cout << "Lens focal length: " << perspective_camera->GetFocalLength() * 1000.f << "mm\n";
            std::cout << "Lens focus distance: " << perspective_camera->GetFocusDistance() << "m\n";
            std::cout << "F-Stop: " << 1.f / (perspective_camera->GetAperture() * 10.f) << "\n";			
        }

        std::cout << "Sensor size: " << settings.camera_sensor_size.x * 1000.f << "x" << settings.camera_sensor_size.y * 1000.f << "mm\n";

		std::cout << "World Box min: " << m_scene->GetWorldAABB().pmin[0] << " " << m_scene->GetWorldAABB().pmin[1] << " " << m_scene->GetWorldAABB().pmin[2] << "\n";
		std::cout << "World Box max: " << m_scene->GetWorldAABB().pmax[0] << " " << m_scene->GetWorldAABB().pmax[1] << " " << m_scene->GetWorldAABB().pmax[2] << "\n";
		
		float3 voxel_extents = m_scene->GetWorldAABB().extents();
		float max_extent = std::max(voxel_extents[0], std::max(voxel_extents[1], voxel_extents[2]));

		m_voxel_data.unit_size = max_extent / settings.voxel_size;
		m_voxel_data.orig = m_scene->GetWorldAABB().pmin;
		m_voxel_data.color.resize(std::pow(settings.voxel_size, 3), float4(0.f));

		std::cout << "Voxel size: "	<< m_voxel_data.color.size() << "\n";
		std::cout << "Voxel unit size: " << m_voxel_data.unit_size << "\n";

		m_voxel_data.color_buffer = m_cfgs[m_primary].context.CreateBuffer<RadeonRays::float4>(m_voxel_data.color.size(), CL_MEM_READ_WRITE);
		static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->SetVoxelCreated(settings.voxel_created);

		std::cout << "Mipmap Level: " << log2f(settings.voxel_size) << "\n";
		for (int i = 0; i <= log2f(settings.voxel_size); i++) {
			CLWBuffer<float4> mipmap_buffer = m_cfgs[m_primary].context.CreateBuffer<RadeonRays::float4>(std::pow(settings.voxel_size/ std::pow(2, i), 3), CL_MEM_READ_WRITE);
			m_voxel_data.mipmap_buffers.push_back(mipmap_buffer);
		}
		std::cout << "Mipmap Level: " << m_voxel_data.mipmap_buffers.size() << "\n";
		cl_int errNum;
		cl_image_format clImageFormat;
		clImageFormat.image_channel_order = CL_RGBA;
		clImageFormat.image_channel_data_type = CL_FLOAT;

		cl_image_desc clImageDesc;
		clImageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
		clImageDesc.image_width = settings.voxel_size;
		clImageDesc.image_height = settings.voxel_size;
		clImageDesc.image_depth = settings.voxel_size;
		clImageDesc.image_row_pitch = 0;
		clImageDesc.image_slice_pitch = 0;
		clImageDesc.num_mip_levels = m_voxel_data.mipmap_buffers.size();
		clImageDesc.num_samples = 0;
		clImageDesc.buffer = NULL;	


		m_voxel_data.voxel_texture = clCreateImage(m_cfgs[m_primary].context, CL_MEM_READ_WRITE, &clImageFormat, &clImageDesc, nullptr, &errNum);	
		if (errNum != CL_SUCCESS)
			std::cerr << getErrorString(errNum) << std::endl;

    }

    void AppClRender::UpdateScene()
    {

        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            if (i == m_primary)
            {
                m_cfgs[i].controller->CompileScene(m_scene);
                m_cfgs[i].renderer->Clear(float3(0, 0, 0), *m_outputs[i].output);
				m_cfgs[i].renderer->Clear(float3(0, 0, 0), *m_outputs[i].output_position);
				m_cfgs[i].renderer->Clear(float3(0, 0, 0), *m_outputs[i].output_normal);
				m_cfgs[i].renderer->Clear(float3(0, 0, 0), *m_outputs[i].output_albedo);

#ifdef ENABLE_DENOISER
                m_cfgs[i].renderer->Clear(float3(0, 0, 0), *m_outputs[i].output_normal);
                m_cfgs[i].renderer->Clear(float3(0, 0, 0), *m_outputs[i].output_position);
                m_cfgs[i].renderer->Clear(float3(0, 0, 0), *m_outputs[i].output_albedo);
                m_cfgs[i].renderer->Clear(float3(0, 0, 0), *m_outputs[i].output_mesh_id);
#endif

            }
            else
                m_ctrl[i].clear.store(true);
        }
    }
	
    void AppClRender::Update(AppSettings& settings)
    {
        //if (std::chrono::duration_cast<std::chrono::seconds>(time - updatetime).count() > 1)
        //{
        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            if (m_cfgs[i].type == ConfigManager::kPrimary)
                continue;

            int desired = 1;
            if (std::atomic_compare_exchange_strong(&m_ctrl[i].newdata, &desired, 0))
            {
                {
                    m_cfgs[m_primary].context.WriteBuffer(0, m_outputs[m_primary].copybuffer, &m_outputs[i].fdata[0], settings.width * settings.height);
                }

                auto acckernel = static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->GetAccumulateKernel();

                int argc = 0;
                acckernel.SetArg(argc++, m_outputs[m_primary].copybuffer);
                acckernel.SetArg(argc++, settings.width * settings.width);
                acckernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(m_outputs[m_primary].output.get())->data());

                int globalsize = settings.width * settings.height;
                m_cfgs[m_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, acckernel);				
            }
        }

        //updatetime = time;
        //}

        if (!settings.interop)
        {
#ifdef ENABLE_DENOISER
            m_outputs[m_primary].output_denoised->GetData(&m_outputs[m_primary].fdata[0]);
#else
            m_outputs[m_primary].output->GetData(&m_outputs[m_primary].fdata[0]);
#endif

            float gamma = 2.2f;
            for (int i = 0; i < (int)m_outputs[m_primary].fdata.size(); ++i)
            {
                m_outputs[m_primary].udata[4 * i] = (unsigned char)clamp(clamp(pow(m_outputs[m_primary].fdata[i].x / m_outputs[m_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
                m_outputs[m_primary].udata[4 * i + 1] = (unsigned char)clamp(clamp(pow(m_outputs[m_primary].fdata[i].y / m_outputs[m_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
                m_outputs[m_primary].udata[4 * i + 2] = (unsigned char)clamp(clamp(pow(m_outputs[m_primary].fdata[i].z / m_outputs[m_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
                m_outputs[m_primary].udata[4 * i + 3] = 1;
            }
			
			
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_outputs[m_primary].output->width(), m_outputs[m_primary].output->height(), GL_RGBA, GL_UNSIGNED_BYTE, &m_outputs[m_primary].udata[0]);			
            glBindTexture(GL_TEXTURE_2D, 0);
			
			
        }
        else
        {
            std::vector<cl_mem> objects;
            objects.push_back(m_cl_interop_image);
            m_cfgs[m_primary].context.AcquireGLObjects(0, objects);

            auto copykernel = static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->GetCopyKernel();
			auto voxelkernel = static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->GetVoxelComputeKernel();
			auto visualizationkernel = static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->GetVoxelVisualizationKernel();
			auto mipmapkernel = static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->GetVoxelMipmapKernel();
			auto voxelconetracingkernel = static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->GetVoxelConeTracingKernel();
			
#ifdef ENABLE_DENOISER
            auto output = m_outputs[m_primary].output_denoised.get();
#else
            auto output = m_outputs[m_primary].output.get();
			auto position_output = m_outputs[m_primary].output_position.get();
			auto normal_output = m_outputs[m_primary].output_normal.get();
			auto albedo_output = m_outputs[m_primary].output_albedo.get();
#endif
			
            int argc;			
			int globalsize;
			if (settings.voxel_mode == 0) {
				argc = 0;
				copykernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(output)->data());
				copykernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(position_output)->data());
				copykernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(normal_output)->data());
				copykernel.SetArg(argc++, output->width());
				copykernel.SetArg(argc++, output->height());
				copykernel.SetArg(argc++, 2.2f);
				copykernel.SetArg(argc++, m_cl_interop_image);

				globalsize = output->width() * output->height();

				m_cfgs[m_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, copykernel);
			}

			if ((settings.samplecount == 30 && !settings.voxel_created && settings.voxel_enabled) || settings.voxel_catch) {

				//m_voxel_data.color.resize(std::pow(settings.voxel_size, 3), float4(0.f));
				m_cfgs[m_primary].context.WriteBuffer(0, m_voxel_data.color_buffer, &m_voxel_data.color[0], m_voxel_data.color.size());

				argc = 0;

				voxelkernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(output)->data());
				voxelkernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(position_output)->data());
				voxelkernel.SetArg(argc++, m_voxel_data.color_buffer);
				voxelkernel.SetArg(argc++, m_voxel_data.orig[0]);
				voxelkernel.SetArg(argc++, m_voxel_data.orig[1]);
				voxelkernel.SetArg(argc++, m_voxel_data.orig[2]);
				voxelkernel.SetArg(argc++, m_voxel_data.unit_size);
				voxelkernel.SetArg(argc++, settings.voxel_size);
				voxelkernel.SetArg(argc++, output->width());
				voxelkernel.SetArg(argc++, output->height());

				globalsize = output->width() * output->height();

				m_cfgs[m_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, voxelkernel);
				m_cfgs[m_primary].context.ReadBuffer(0, m_voxel_data.color_buffer, &m_voxel_data.color[0], m_voxel_data.color.size());
				
				settings.voxel_sample_count++;
				
				if (settings.voxel_sample_count == settings.voxel_sample) {
					settings.voxel_created = 1;
					m_camera->LookAt(settings.camera_pos
						, settings.camera_at
						, settings.camera_up);
					static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->SetVoxelCreated(settings.voxel_created);
				}
				UpdateScene();
				settings.samplecount = 0;
				if (settings.voxel_catch) {
					settings.voxel_catch = 0;
					settings.voxel_created = 1;
					settings.voxel_enabled = 1;
					settings.voxel_mipmaped = 0;
				}
				
			}
			if(!settings.voxel_created && !settings.voxel_enabled){
				settings.voxel_created = 1;
				m_camera->LookAt(settings.camera_pos
					, settings.camera_at
					, settings.camera_up);
				static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->SetVoxelCreated(settings.voxel_created);
			}
				
			if (settings.voxel_created && settings.voxel_enabled) {
				
				if (!settings.voxel_mipmaped) {
					m_cfgs[m_primary].context.WriteBuffer(0, m_voxel_data.color_buffer, &m_voxel_data.color[0], m_voxel_data.color.size());
					for (int i = 0; i < m_voxel_data.mipmap_buffers.size(); i++) {
						//std::cout << "Create Mipmap Level: " << i << "\n";
						argc = 0;
						if (i == 0) {
							mipmapkernel.SetArg(argc++, m_voxel_data.color_buffer);
							mipmapkernel.SetArg(argc++, settings.voxel_size);
						}
						else {
							mipmapkernel.SetArg(argc++, m_voxel_data.mipmap_buffers[i - 1]);
							mipmapkernel.SetArg(argc++, (int)(settings.voxel_size / std::pow(2, i - 1)));
						}
						mipmapkernel.SetArg(argc++, m_voxel_data.mipmap_buffers[i]);
						mipmapkernel.SetArg(argc++, (int)(settings.voxel_size / std::pow(2, i)));
						mipmapkernel.SetArg(argc++, i);

						globalsize = std::pow(settings.voxel_size / std::pow(2, i), 3);
						//std::cout << globalsize << "\n";

						m_cfgs[m_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, mipmapkernel);

					}
					settings.voxel_mipmaped = 1;
				}
				if (settings.voxel_mipmaped && settings.voxel_mode > 1) {

					argc = 0;

					voxelconetracingkernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(output)->data());
					voxelconetracingkernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(position_output)->data());
					voxelconetracingkernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(normal_output)->data());
					voxelconetracingkernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(albedo_output)->data());
					voxelconetracingkernel.SetArg(argc++, log2f(settings.voxel_size));
					voxelconetracingkernel.SetArg(argc++, m_voxel_data.orig[0]);
					voxelconetracingkernel.SetArg(argc++, m_voxel_data.orig[1]);
					voxelconetracingkernel.SetArg(argc++, m_voxel_data.orig[2]);
					voxelconetracingkernel.SetArg(argc++, sizeof(cl_float3), &settings.camera_pos);
					for (int i = 0; i < 9; i++) {
						if (i < m_voxel_data.mipmap_buffers.size()) {
							voxelconetracingkernel.SetArg(argc++, m_voxel_data.mipmap_buffers[i]);
							voxelconetracingkernel.SetArg(argc++, float(m_voxel_data.unit_size * pow(2, i)));
							voxelconetracingkernel.SetArg(argc++, int(settings.voxel_size / pow(2, i)));
						}
						else {

							voxelconetracingkernel.SetArg(argc++, nullptr);
							voxelconetracingkernel.SetArg(argc++, 0.f);
							voxelconetracingkernel.SetArg(argc++, 0);
						}
					}
					voxelconetracingkernel.SetArg(argc++, settings.voxel_mode);
					voxelconetracingkernel.SetArg(argc++, output->width());
					voxelconetracingkernel.SetArg(argc++, output->height());
					voxelconetracingkernel.SetArg(argc++, 2.2f);
					voxelconetracingkernel.SetArg(argc++, m_cl_interop_image);

					globalsize = output->width() * output->height();

					m_cfgs[m_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, voxelconetracingkernel);
				}

				if(settings.voxel_mode == 1){
					//Voxel Visualization
					argc = 0;
					int mipmap_level = settings.voxel_mipmap_level;
					float level_unit = m_voxel_data.unit_size * pow(2, mipmap_level);
					int level_voxel_size = settings.voxel_size / pow(2, mipmap_level);
					//std::cout << level_unit << " " << level_voxel_size << "\n";
					
					visualizationkernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(position_output)->data());
					visualizationkernel.SetArg(argc++, m_voxel_data.mipmap_buffers[mipmap_level]);
					visualizationkernel.SetArg(argc++, m_voxel_data.orig[0]);
					visualizationkernel.SetArg(argc++, m_voxel_data.orig[1]);
					visualizationkernel.SetArg(argc++, m_voxel_data.orig[2]);
					visualizationkernel.SetArg(argc++, level_unit);
					visualizationkernel.SetArg(argc++, level_voxel_size);
					visualizationkernel.SetArg(argc++, output->width());
					visualizationkernel.SetArg(argc++, output->height());
					visualizationkernel.SetArg(argc++, 2.2f);
					visualizationkernel.SetArg(argc++, m_cl_interop_image);

					globalsize = output->width() * output->height();

					m_cfgs[m_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, visualizationkernel);
				}
			}		

            m_cfgs[m_primary].context.ReleaseGLObjects(0, objects);
            m_cfgs[m_primary].context.Finish(0);
        }


        if (settings.benchmark)
        {
            auto& scene = m_cfgs[m_primary].controller->CompileScene(m_scene);
            static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->Benchmark(scene, settings.stats);

            settings.benchmark = false;
            settings.rt_benchmarked = true;
        }

        //ClwClass::Update();
    }

	void AppClRender::Render(int sample_cnt)
	{
#ifdef ENABLE_DENOISER
		WaveletDenoiser* wavelet_denoiser = dynamic_cast<WaveletDenoiser*>(m_outputs[m_primary].denoiser.get());

		if (wavelet_denoiser != nullptr)
		{
			wavelet_denoiser->Update(static_cast<PerspectiveCamera*>(m_camera.get()));
		}
#endif
		auto& scene = m_cfgs[m_primary].controller->GetCachedScene(m_scene);		
		m_cfgs[m_primary].renderer->Render(scene);

		if (m_shape_id_requested)
		{
			// offset in OpenCl memory till necessary item
			auto offset = (std::uint32_t)(m_width * (m_height - m_shape_id_pos.y) + m_shape_id_pos.x);
			// copy shape id elem from OpenCl
			float4 shape_id;
			m_shape_id_data.output->GetData((float3*)&shape_id, offset, 1);
			m_promise.set_value(shape_id.x);
			// clear output to stop tracking shape id map in openCl
			m_cfgs[m_primary].renderer->SetOutput(Renderer::OutputType::kShapeId, nullptr);
			m_shape_id_requested = false;
		}
		

#ifdef ENABLE_DENOISER
        Baikal::PostEffect::InputSet input_set;
        input_set[Baikal::Renderer::OutputType::kColor] = m_outputs[m_primary].output.get();
        input_set[Baikal::Renderer::OutputType::kWorldShadingNormal] = m_outputs[m_primary].output_normal.get();
        input_set[Baikal::Renderer::OutputType::kWorldPosition] = m_outputs[m_primary].output_position.get();
        input_set[Baikal::Renderer::OutputType::kAlbedo] = m_outputs[m_primary].output_albedo.get();
        input_set[Baikal::Renderer::OutputType::kMeshID] = m_outputs[m_primary].output_mesh_id.get();

        auto radius = 10U - RadeonRays::clamp((sample_cnt / 16), 1U, 9U);
        auto position_sensitivity = 5.f + 10.f * (radius / 10.f);

        const bool is_bilateral_denoiser = dynamic_cast<BilateralDenoiser*>(m_outputs[m_primary].denoiser.get()) != nullptr;

        if (is_bilateral_denoiser)
        {

            auto normal_sensitivity = 0.1f + (radius / 10.f) * 0.15f;
            auto color_sensitivity = (radius / 10.f) * 2.f;
            auto albedo_sensitivity = 0.5f + (radius / 10.f) * 0.5f;
            m_outputs[m_primary].denoiser->SetParameter("radius", radius);
            m_outputs[m_primary].denoiser->SetParameter("color_sensitivity", color_sensitivity);
            m_outputs[m_primary].denoiser->SetParameter("normal_sensitivity", normal_sensitivity);
            m_outputs[m_primary].denoiser->SetParameter("position_sensitivity", position_sensitivity);
            m_outputs[m_primary].denoiser->SetParameter("albedo_sensitivity", albedo_sensitivity);
        }

        m_outputs[m_primary].denoiser->Apply(input_set, *m_outputs[m_primary].output_denoised);
#endif
    }

	void AppClRender::SaveVoxelData(std::string modelname) {
		std::fstream fs;
		fs.open(modelname, std::ios::out);
		if (!fs)
			std::cout << "Voxeldata save failed!\n";
		std::cout << "Voxeldata saving...\n";
		fs << "s " << m_voxel_data.color.size() << "\n";
		for (size_t i = 0; i < m_voxel_data.color.size(); i++)
		{
			fs << "v " << m_voxel_data.color[i][0] << " " << m_voxel_data.color[i][1] << " " << m_voxel_data.color[i][2] << " " << m_voxel_data.color[i][3] << "\n";
		}
		fs.close();
		std::cout << "Voxeldata save completed!\n";
	}

	void AppClRender::LoadVoxelData(std::string dataname) {
		std::ifstream ifile(dataname);
		std::string temp;
		int count = 0;
		int size;
		while (ifile >> temp) {
			switch (temp[0]) {
			case 's':
				ifile >> size;
				std::cout << size << "\nVoxeldata loading...\n";
				if (size != m_voxel_data.color.size()) {
					std::cout << "Voxeldata in diffrent size !\n";
					ifile.close();
					return;
				}
				break;
			case 'v':
				ifile >> m_voxel_data.color[count][0] >> m_voxel_data.color[count][1] >> m_voxel_data.color[count][2] >> m_voxel_data.color[count][3];
				count++;
				break;
			}
		}
		std::cout << "Voxeldata load completed!\n";
		ifile.close();
	}

    void AppClRender::SaveFrameBuffer(AppSettings& settings)
    {
        std::vector<RadeonRays::float3> data;

        //read cl output in case of iterop
        std::vector<RadeonRays::float3> output_data;
        if (settings.interop)
        {
            auto output = m_outputs[m_primary].output.get();
            auto buffer = static_cast<Baikal::ClwOutput*>(output)->data();
            output_data.resize(buffer.GetElementCount());
            m_cfgs[m_primary].context.ReadBuffer(0, static_cast<Baikal::ClwOutput*>(output)->data(), &output_data[0], output_data.size()).Wait();
        }

        //use already copied to CPU cl data in case of no interop
        auto& fdata = settings.interop ? output_data : m_outputs[m_primary].fdata;

        data.resize(fdata.size());
        std::transform(fdata.cbegin(), fdata.cend(), data.begin(),
            [](RadeonRays::float3 const& v)
        {
            float invw = 1.f / v.w;
            return v * invw;
        });

        std::stringstream oss;
        auto camera_position = m_camera->GetPosition();
        auto camera_direction = m_camera->GetForwardVector();
        oss << "../Output/" << settings.modelname << "_p" << camera_position.x << camera_position.y << camera_position.z <<
            "_d" << camera_direction.x << camera_direction.y << camera_direction.z <<
            "_s" << settings.num_samples << ".exr";

        SaveImage(oss.str(), settings.width, settings.height, data.data());
    }

    void AppClRender::SaveImage(const std::string& name, int width, int height, const RadeonRays::float3* data)
    {
        OIIO_NAMESPACE_USING;

        std::vector<float3> tempbuf(width * height);
        tempbuf.assign(data, data + width * height);

        for (auto y = 0; y < height; ++y)
            for (auto x = 0; x < width; ++x)
            {

                float3 val = data[(height - 1 - y) * width + x];
                tempbuf[y * width + x] = (1.f / val.w) * val;

                tempbuf[y * width + x].x = std::pow(tempbuf[y * width + x].x, 1.f / 2.2f);
                tempbuf[y * width + x].y = std::pow(tempbuf[y * width + x].y, 1.f / 2.2f);
                tempbuf[y * width + x].z = std::pow(tempbuf[y * width + x].z, 1.f / 2.2f);
            }

        ImageOutput* out = ImageOutput::create(name);

        if (!out)
        {
            throw std::runtime_error("Can't create image file on disk");
        }

        ImageSpec spec(width, height, 3, TypeDesc::FLOAT);

        out->open(name, spec);
        out->write_image(TypeDesc::FLOAT, &tempbuf[0], sizeof(float3));
        out->close();
    }

    void AppClRender::RenderThread(ControlData& cd)
    {
        auto renderer = m_cfgs[cd.idx].renderer.get();
        auto controller = m_cfgs[cd.idx].controller.get();
        auto output = m_outputs[cd.idx].output.get();

        auto updatetime = std::chrono::high_resolution_clock::now();

        while (!cd.stop.load())
        {
            int result = 1;
            bool update = false;

            if (std::atomic_compare_exchange_strong(&cd.clear, &result, 0))
            {
                renderer->Clear(float3(0, 0, 0), *output);
                controller->CompileScene(m_scene);
                update = true;
            }

            auto& scene = controller->GetCachedScene(m_scene);
            renderer->Render(scene);

            auto now = std::chrono::high_resolution_clock::now();

            update = update || (std::chrono::duration_cast<std::chrono::seconds>(now - updatetime).count() > 1);

            if (update)
            {
                m_outputs[cd.idx].output->GetData(&m_outputs[cd.idx].fdata[0]);
                updatetime = now;
                cd.newdata.store(1);
            }

            m_cfgs[cd.idx].context.Finish(0);
        }
    }

    void AppClRender::StartRenderThreads()
    {
        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            if (i != m_primary)
            {
                m_renderthreads.push_back(std::thread(&AppClRender::RenderThread, this, std::ref(m_ctrl[i])));
                m_renderthreads.back().detach();
            }
        }

        std::cout << m_cfgs.size() << " OpenCL submission threads started\n";
    }

    void AppClRender::StopRenderThreads()
    {
        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            if (i == m_primary)
                continue;

            m_ctrl[i].stop.store(true);
        }
    }

    void AppClRender::RunBenchmark(AppSettings& settings)
    {
        std::cout << "Running general benchmark...\n";

        auto time_bench_start_time = std::chrono::high_resolution_clock::now();
        for (auto i = 0U; i < 512; ++i)
        {
            Render(0);
        }

        m_cfgs[m_primary].context.Finish(0);

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::high_resolution_clock::now() - time_bench_start_time).count();

        settings.time_benchmark_time = delta / 1000.f;

        m_outputs[m_primary].output->GetData(&m_outputs[m_primary].fdata[0]);
        float gamma = 2.2f;
        for (int i = 0; i < (int)m_outputs[m_primary].fdata.size(); ++i)
        {
            m_outputs[m_primary].udata[4 * i] = (unsigned char)clamp(clamp(pow(m_outputs[m_primary].fdata[i].x / m_outputs[m_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
            m_outputs[m_primary].udata[4 * i + 1] = (unsigned char)clamp(clamp(pow(m_outputs[m_primary].fdata[i].y / m_outputs[m_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
            m_outputs[m_primary].udata[4 * i + 2] = (unsigned char)clamp(clamp(pow(m_outputs[m_primary].fdata[i].z / m_outputs[m_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
            m_outputs[m_primary].udata[4 * i + 3] = 1;
        }

        auto& fdata = m_outputs[m_primary].fdata;
        std::vector<RadeonRays::float3> data(fdata.size());
        std::transform(fdata.cbegin(), fdata.cend(), data.begin(),
            [](RadeonRays::float3 const& v)
        {
            float invw = 1.f / v.w;
            return v * invw;
        });

        std::stringstream oss;
        oss << "../Output/" << settings.modelname << ".exr";

        SaveImage(oss.str(), settings.width, settings.height, &data[0]);

        std::cout << "Running RT benchmark...\n";

        auto& scene = m_cfgs[m_primary].controller->GetCachedScene(m_scene);
        static_cast<MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->Benchmark(scene, settings.stats);
    }

    void AppClRender::SetNumBounces(int num_bounces)
    {
        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[i].renderer.get())->SetMaxBounces(num_bounces);
        }
    }

    void AppClRender::SetOutputType(Renderer::OutputType type)
    {
        for (int i = 0; i < m_cfgs.size(); ++i)
        {
            m_cfgs[i].renderer->SetOutput(m_output_type, nullptr);
            m_cfgs[i].renderer->SetOutput(type, m_outputs[i].output.get());
        }
        m_output_type = type;
    }


    std::future<int> AppClRender::GetShapeId(std::uint32_t x, std::uint32_t y)
    {
        m_promise = std::promise<int>();
        if (x >= m_width || y >= m_height)
            throw std::logic_error(
                "AppClRender::GetShapeId(...): x or y cords beyond the size of image");

        if (m_cfgs.empty())
            throw std::runtime_error("AppClRender::GetShapeId(...): config vector is empty");

        // enable aov shape id output from OpenCl
        m_cfgs[m_primary].renderer->SetOutput(
            Renderer::OutputType::kShapeId, m_shape_id_data.output.get());
        m_shape_id_pos = RadeonRays::float2((float)x, (float)y);
        // request shape id from render
        m_shape_id_requested = true;
        return m_promise.get_future();
    }

    Baikal::Shape::Ptr AppClRender::GetShapeById(int shape_id)
    {
        if (shape_id < 0)
            return nullptr;

        // find shape in scene by its id
        for (auto iter = m_scene->CreateShapeIterator(); iter->IsValid(); iter->Next())
        {
            auto shape = iter->ItemAs<Shape>();
            if (shape->GetId() == shape_id)
                return shape;
        }
        return nullptr;
    }

#ifdef ENABLE_DENOISER
    void AppClRender::SetDenoiserFloatParam(const std::string& name, const float4& value)
    {
        m_outputs[m_primary].denoiser->SetParameter(name, value);
    }

    float4 AppClRender::GetDenoiserFloatParam(const std::string& name)
    {
        return m_outputs[m_primary].denoiser->GetParameter(name);
    }
#endif
} // Baikal
