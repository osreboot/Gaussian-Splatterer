#pragma once

#include <chrono>
#include <wx/wx.h>
#include <wx/aui/framemanager.h>

class Project;
class RtxHost;
class Trainer;

class UiPanelViewInput;
class UiPanelViewOutput;

class UiPanelParamsLr;
class UiPanelParamsDensify;
class UiPanelParamsOther;

class UiPanelToolsInput;
class UiPanelToolsTrain;
class UiPanelToolsTruth;
class UiPanelToolsView;

class UiFrame : public wxFrame {

private:
    std::chrono::high_resolution_clock::time_point timeLastUpdate, timeNow;

    wxAuiManager auiManager;
    wxString auiPersDefault;

    float autoTrainingBudget = 0.0f; // Number of allowed training steps per second

public:
    Project* project = nullptr;

    RtxHost* rtx;
    Trainer* trainer;

    wxMenuBar* menuBar;
    wxMenu* menuFile;
    wxMenu* menuFileNew;
    wxMenu* menuFileLoad;
    wxMenu* menuFileSave;
    wxMenu* menuView;
    wxMenu* menuAbout;

    UiPanelViewInput* panelInput;
    UiPanelViewOutput* panelOutput;

    UiPanelParamsLr* panelParamsLr;
    UiPanelParamsDensify* panelParamsDensify;
    UiPanelParamsOther* panelParamsOther;

    UiPanelToolsInput* panelToolsInput;
    UiPanelToolsTruth* panelToolsTruth;
    UiPanelToolsTrain* panelToolsTrain;
    UiPanelToolsView* panelToolsView;

    bool autoTraining = false;

    UiFrame();
    ~UiFrame() override;

private:
    void initProject(); // Reset to a new project
    void initFieldGrid(); // Initialize splats with a scene-sized grid
    void initFieldMono(); // Initialize splats with a single giant splat
    void initFieldModel(); // Initialize splats with one splat per model triangle

    void update();

    void refreshProject(); // Called when the project data gets changed, update all text fields & spinners

    void saveSettings(const std::string& path);
    void saveSplats(const std::string& path);

    void loadSettings(const std::string& path);
    void loadSplats(const std::string& path);

    void onMenuButton(wxCommandEvent& event);

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

    static constexpr std::string_view NAME_VIEW_INPUT = "VIEW_INPUT";
    static constexpr std::string_view NAME_VIEW_OUTPUT = "VIEW_OUTPUT";
    static constexpr std::string_view NAME_PARAMS_LR = "PARAMS_LR";
    static constexpr std::string_view NAME_PARAMS_DENSIFY = "PARAMS_DENSIFY";
    static constexpr std::string_view NAME_PARAMS_OTHER = "PARAMS_OTHER";
    static constexpr std::string_view NAME_TOOLS_INPUT = "TOOLS_INPUT";
    static constexpr std::string_view NAME_TOOLS_TRUTH = "TOOLS_TRUTH";
    static constexpr std::string_view NAME_TOOLS_TRAIN = "TOOLS_TRAIN";
    static constexpr std::string_view NAME_TOOLS_VIEW = "TOOLS_VIEW";

    enum MenuIds {
        FILE_NEW_PROJECT,
        FILE_NEW_FIELD_GRID,
        FILE_NEW_FIELD_MONO,
        FILE_NEW_FIELD_MODEL,
        FILE_SAVE_PROJECT,
        FILE_SAVE_SPLATS,
        FILE_SAVE_SETTINGS,
        FILE_LOAD_PROJECT,
        FILE_LOAD_SPLATS,
        FILE_LOAD_SETTINGS,
        VIEW_PERS_DEFAULT,
        ABOUT_ABOUT
    };

};