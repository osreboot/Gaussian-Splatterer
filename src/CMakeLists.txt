target_sources(${PROJECT_NAME} PRIVATE
        Main.cpp
        Camera.cpp
        Camera.h
        Config.h
        FboRenderer.cpp
        FboRenderer.h
        ModelSplatsDevice.cpp
        ModelSplatsDevice.h
        ModelSplatsHost.cpp
        ModelSplatsHost.h
        OpenGLIncludes.h
        Project.h
        StbImpl.cpp
        Trainer.cu
        Trainer.cuh
        rtx/RtxHost.cpp
        rtx/RtxHost.h
        ui/dialog/UiDialogAbout.cpp
        ui/dialog/UiDialogAbout.h
        ui/params/UiPanelParamsDensify.cpp
        ui/params/UiPanelParamsDensify.h
        ui/params/UiPanelParamsLr.cpp
        ui/params/UiPanelParamsLr.h
        ui/params/UiPanelParamsOther.cpp
        ui/params/UiPanelParamsOther.h
        ui/tools/UiPanelToolsInput.cpp
        ui/tools/UiPanelToolsInput.h
        ui/tools/UiPanelToolsTrain.cpp
        ui/tools/UiPanelToolsTrain.h
        ui/tools/UiPanelToolsTruth.cpp
        ui/tools/UiPanelToolsTruth.h
        ui/tools/UiPanelToolsView.cpp
        ui/tools/UiPanelToolsView.h
        ui/UiFrame.cpp
        ui/UiFrame.h
        ui/UiPanelViewInput.cpp
        ui/UiPanelViewInput.h
        ui/UiPanelViewOutput.cpp
        ui/UiPanelViewOutput.h
        )