#pragma once

#include <chrono>
#include <wx/wx.h>

class Project;
class RtxHost;
class Trainer;

class UiPanelViewInput;
class UiPanelViewOutput;
class UiPanelParams;
class UiPanelTools;

class UiFrame : public wxFrame {

private:
    std::chrono::high_resolution_clock::time_point timeLastUpdate, timeNow;

    wxPanel* panel;
    wxBoxSizer* sizer;
    wxBoxSizer* sizerViews;

    float autoTrainingBudget = 0.0f;

public:
    Project* project = nullptr;

    RtxHost* rtx;
    Trainer* trainer;

    wxMenuBar* menuBar;
    wxMenu* menuFile;
    wxMenu* menuFileNew;
    wxMenu* menuFileLoad;
    wxMenu* menuFileSave;
    wxMenu* menuAbout;

    UiPanelViewInput* panelInput;
    UiPanelViewOutput* panelOutput;
    UiPanelParams* panelParams;
    UiPanelTools* panelTools;

    bool autoTraining = false;

    UiFrame();
    ~UiFrame() override;

private:
    void initProject();
    void initFieldGrid();
    void initFieldMono();
    void initFieldModel();

    void update();

    void refreshProject();

    void saveSettings(const std::string& path) const;
    void saveSplats(const std::string& path) const;

    void loadSettings(const std::string& path);
    void loadSplats(const std::string& path);

    void onMenuButton(wxCommandEvent& event);

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

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
        ABOUT_ABOUT
    };

};