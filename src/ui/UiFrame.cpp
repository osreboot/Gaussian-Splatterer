#include "UiFrame.h"

#include <fstream>
#include <wx/progdlg.h>

#include "UiPanelViewInput.h"
#include "UiPanelViewOutput.h"
#include "UiPanelParams.h"
#include "UiPanelTools.h"
#include "dialog/UiDialogAbout.h"

#include "Project.h"
#include "ModelSplatsDevice.h"
#include "ModelSplatsHost.h"
#include "rtx/RtxHost.h"
#include "Trainer.cuh"

using namespace std;

UiFrame::UiFrame() :
        wxFrame(nullptr, wxID_ANY, "Gaussian Splatterer") {
    panel = new wxPanel(this);
    sizer = new wxBoxSizer(wxVERTICAL);
    panel->SetSizerAndFit(sizer);

    initProject();

    rtx = new RtxHost();

    trainer = new Trainer();
    initFieldGrid();

    timeLastUpdate = chrono::high_resolution_clock::now();

    menuBar = new wxMenuBar();
    SetMenuBar(menuBar);

    menuFile = new wxMenu();
    menuFileNew = new wxMenu();
    menuFileLoad = new wxMenu();
    menuFileSave = new wxMenu();

    menuBar->Append(menuFile, "File");
    menuFile->AppendSubMenu(menuFileNew, "New");
    menuFile->AppendSeparator();
    menuFile->AppendSubMenu(menuFileSave, "Save");
    menuFile->AppendSubMenu(menuFileLoad, "Load");
    menuFileNew->Append(FILE_NEW_PROJECT, "New Project");
    menuFileNew->AppendSeparator();
    menuFileNew->Append(FILE_NEW_FIELD_GRID, "New Field From Grid");
    menuFileNew->Append(FILE_NEW_FIELD_MONO, "New Field From Monolithic Splat");
    menuFileNew->Append(FILE_NEW_FIELD_MODEL, "New Field From Model Triangles");
    menuFileSave->Append(FILE_SAVE_PROJECT, "Save Project");
    menuFileSave->AppendSeparator();
    menuFileSave->Append(FILE_SAVE_SPLATS, "Save Splats");
    menuFileSave->Append(FILE_SAVE_SETTINGS, "Save Settings");
    menuFileLoad->Append(FILE_LOAD_PROJECT, "Load Project");
    menuFileLoad->AppendSeparator();
    menuFileLoad->Append(FILE_LOAD_SPLATS, "Load Splats");
    menuFileLoad->Append(FILE_LOAD_SETTINGS, "Load Settings");

    menuAbout = new wxMenu();
    menuBar->Append(menuAbout, "About");
    menuAbout->Append(ABOUT_ABOUT, "About");

    menuBar->Bind(wxEVT_COMMAND_TOOL_CLICKED, &UiFrame::onMenuButton, this);

    sizerViews = new wxBoxSizer(wxHORIZONTAL);
    sizer->Add(sizerViews, wxSizerFlags(1).Expand());

    panelInput = new UiPanelViewInput(panel);
    sizerViews->Add(panelInput, wxSizerFlags(1).Expand().Border());

    panelOutput = new UiPanelViewOutput(panel, panelInput->context);
    sizerViews->Add(panelOutput, wxSizerFlags(1).Expand().Border());

    panelParams = new UiPanelParams(panel);
    sizerViews->Add(panelParams, wxSizerFlags().Border());

    panelTools = new UiPanelTools(panel);
    sizer->Add(panelTools, wxSizerFlags().Border());

    sizer->Fit(this);

    refreshProject();
}

UiFrame::~UiFrame() {
    delete project;

    delete rtx;
    delete trainer;
}

void UiFrame::initProject() {
    delete project;
    project = new Project();
    project->sphere2.fovDeg = 30.0f;
}

void UiFrame::initFieldGrid() {
    ModelSplatsHost modelHost(1000000, 1, 4);

    static const float dim = 4.0f;
    static const float step = 0.5f;

    std::vector<float> shs;
    for(int i = 0; i < 3 * modelHost.shCoeffs; i++) shs.push_back(0.0f);

    for(float x = -dim; x <= dim; x += step){
        for(float y = -dim; y <= dim; y += step){
            for(float z = -dim; z <= dim; z += step){
                modelHost.pushBack({x, y, z}, shs, {step * 0.1f, step * 0.1f, step * 0.1f},
                                   1.0f, glm::angleAxis(0.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
            }
        }
    }

    delete trainer->model;
    trainer->model = new ModelSplatsDevice(modelHost);
    project->iterations = 0;
}

void UiFrame::initFieldMono() {
    ModelSplatsHost modelHost(1000000, 1, 4);

    std::vector<float> shs;
    for(int i = 0; i < 3 * modelHost.shCoeffs; i++) shs.push_back(0.0f);

    modelHost.pushBack({0.0f, 0.0f, 0.0f}, shs, {0.3f, 0.3f, 0.3f}, 1.0f,
                       glm::angleAxis(0.0f, glm::vec3(0.0f, 1.0f, 0.0f)));

    delete trainer->model;
    trainer->model = new ModelSplatsDevice(modelHost);
    project->iterations = 0;
}

void UiFrame::initFieldModel() {
    if (!std::filesystem::exists(project->pathModel)) {
        wxMessageDialog dialog(this, "Failed to load the model file for splat field generation!", "Model Load Failed", wxICON_ERROR);
        dialog.ShowModal();
        return;
    }

    std::vector<owl::vec3f> vertices;
    std::vector<owl::vec3i> triangles;

    std::ifstream ifs(project->pathModel);
    std::string line;
    while(getline(ifs, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if(prefix == "v") { // Vertex definition
            float x, y, z;
            iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        }else if(prefix == "vt") { // Vertex definition
            float u, v;
            iss >> u >> v;
        }else if(prefix == "f") { // Face definition
            // Space-separated vertices list, with the syntax: vertexIndex/vertexTextureIndex/vertexNormalIndex
            std::vector<owl::vec3i> faceVertices;

            std::string sFaceVertex;
            while(iss >> sFaceVertex) {
                owl::vec3i faceVertex = {0, 0, 0};
                int charIndex = 0;
                while(charIndex < sFaceVertex.length() && sFaceVertex[charIndex] != '/'){
                    faceVertex.x = faceVertex.x * 10 + (sFaceVertex[charIndex++] - '0');
                }
                charIndex++;
                while(charIndex < sFaceVertex.length() && sFaceVertex[charIndex] != '/'){
                    faceVertex.y = faceVertex.y * 10 + (sFaceVertex[charIndex++] - '0');
                }
                charIndex++;
                while(charIndex < sFaceVertex.length() && sFaceVertex[charIndex] != '/'){
                    faceVertex.z = faceVertex.z * 10 + (sFaceVertex[charIndex++] - '0');
                }
                faceVertices.push_back(faceVertex);
            }
            if(faceVertices.size() == 4) {
                triangles.push_back(owl::vec3i(faceVertices[0].x, faceVertices[1].x, faceVertices[2].x) - owl::vec3i(1));
                triangles.push_back(owl::vec3i(faceVertices[0].x, faceVertices[2].x, faceVertices[3].x) - owl::vec3i(1));
            }else if(faceVertices.size() == 3) {
                triangles.push_back(owl::vec3i(faceVertices[0].x, faceVertices[1].x, faceVertices[2].x) - owl::vec3i(1));
            }else throw std::runtime_error("Unexpected vertex count in face list!" + std::to_string(faceVertices.size()));
        }
    }

    ModelSplatsHost modelHost(1000000, 1, 4);

    for(owl::vec3i triangle : triangles) {
        glm::vec3 v0(vertices[triangle.x].x, vertices[triangle.x].y, vertices[triangle.x].z);
        glm::vec3 v1(vertices[triangle.y].x, vertices[triangle.y].y, vertices[triangle.y].z);
        glm::vec3 v2(vertices[triangle.z].x, vertices[triangle.z].y, vertices[triangle.z].z);

        glm::vec3 location = (v0 + v1 + v2) / 3.0f;
        glm::vec3 scale(glm::length(v1 - v0), glm::length(v2 - v0), 0.005f);
        scale *= 0.2f;

        std::vector<float> shs;
        for(int i = 0; i < 3 * modelHost.shCoeffs; i++) shs.push_back(0.0f);

        glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);
        glm::vec3 dir = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        glm::vec3 axis = glm::cross(up, dir);
        float angle = glm::acos(glm::dot(up, dir));

        modelHost.pushBack(location, shs, scale, 1.0f, glm::angleAxis(angle, axis));
    }

    delete trainer->model;
    trainer->model = new ModelSplatsDevice(modelHost);
    project->iterations = 0;
}

void UiFrame::update() {
    timeNow = chrono::high_resolution_clock::now();
    float delta = (float)chrono::duration_cast<chrono::nanoseconds>(timeNow - timeLastUpdate).count() / 1000000000.0f;
    //delta = min(delta, 0.2f);
    timeLastUpdate = timeNow;

    project->previewTimer += delta;

    if(autoTraining) {
        autoTrainingBudget = min(1.0f, autoTrainingBudget + delta * 100.0f);

        if(autoTrainingBudget >= 1.0f) {
            autoTrainingBudget = 0.0f;

            const bool capture = project->intervalCapture > 0 && project->iterations % project->intervalCapture == 0;
            const bool densify = project->intervalDensify > 0 && project->iterations % project->intervalDensify == 0;

            if(capture) {
                wxCommandEvent eventFake = wxCommandEvent(wxEVT_NULL, 0);
                panelTools->panelTruth->onButtonRandomRotate(eventFake);
                panelTools->panelTruth->onButtonCapture(eventFake);
            }
            trainer->train(*project, densify);
            panelOutput->refreshText();
            panelTools->panelTrain->refreshText();
        }
    }
}

void UiFrame::refreshProject() {
    rtx->reset();

    // TODO nuke trainer captures

    // TODO stop auto training
    // TODO stop auto training in nested classes

    panelInput->refreshProject();
    panelOutput->refreshProject();
    panelParams->refreshProject();
    panelTools->refreshProject();
}

void UiFrame::saveSettings(const std::string& path) const {
    nlohmann::json j;
    nlohmann::to_json(j, *project);

    std::ofstream file(path);
    file << j;
}

void UiFrame::saveSplats(const std::string& path) const {
    wxProgressDialog dialog("Saving Gaussian Splats", "Writing splats to \"" + path + "\"...", trainer->model->count + 1000, panel, wxPD_AUTO_HIDE);
    ModelSplatsHost model(*trainer->model);

    int progress = 1000;
    dialog.Update(progress);

    std::ofstream file(path);
    for (int i = 0; i < model.count; i++) {
        file << "v " << model.locations[i * 3] << " " << model.locations[i * 3 + 1] << " " << model.locations[i * 3 + 2] << "\n";

        file << "sh";
        for (int f = 0; f < model.shCoeffs * 3; f++) {
            file << " " << model.shs[i * 3 * model.shCoeffs + f];
        }
        file << "\n";

        file << "s " << model.scales[i * 3] << " " << model.scales[i * 3 + 1] << " " << model.scales[i * 3 + 2] << "\n";
        file << "a " << model.opacities[i] << "\n";
        file << "r " << model.rotations[i * 4] << " " << model.rotations[i * 4 + 1] << " " << model.rotations[i * 4 + 2] << " " << model.rotations[i * 4 + 3] << "\n";

        dialog.Update(++progress);
    }
}

void UiFrame::loadSettings(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        wxMessageDialog dialog(this, "Failed to load settings file at \"" + path + "\"!", "Load Failed!", wxICON_ERROR);
        dialog.ShowModal();
        return;
    }

    nlohmann::json j;
    std::ifstream file(path);
    file >> j;
    nlohmann::from_json(j, *project);
}

void UiFrame::loadSplats(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        wxMessageDialog dialog(this, "Failed to load splats file at \"" + path + "\"!", "Load Failed!", wxICON_ERROR);
        dialog.ShowModal();
        return;
    }

    // Count the number of lines in the file so we can get a progress estimate
    std::ifstream fileLines(path);
    int linesCount = (int)std::count(std::istreambuf_iterator<char>(fileLines), std::istreambuf_iterator<char>(), '\n');

    wxProgressDialog dialog("Loading Gaussian Splats", "Loading splats from \"" + path + "\"...", linesCount + 2000, panel, wxPD_AUTO_HIDE);

    int progress = 0;

    std::optional<int> shCoeffs = nullopt;

    std::vector<float> locations;
    std::vector<float> shs;
    std::vector<float> scales;
    std::vector<float> opacities;
    std::vector<float> rotations;

    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            for (int f = 0; f < 3; f++) {
                float x;
                iss >> x;
                locations.push_back(x);
            }
        } else if (prefix == "sh") {
            int shCoeffsCount = 0;
            float x;
            while (iss >> x) {
                shs.push_back(x);
                shCoeffsCount++;
            }
            if (!shCoeffs) shCoeffs = shCoeffsCount;
            else assert(shCoeffs == shCoeffsCount);
        } else if (prefix == "s") {
            for (int f = 0; f < 3; f++) {
                float x;
                iss >> x;
                scales.push_back(x);
            }
        } else if (prefix == "a") {
            float x;
            iss >> x;
            opacities.push_back(x);
        } else if (prefix == "r") {
            for (int f = 0; f < 4; f++) {
                float x;
                iss >> x;
                rotations.push_back(x);
            }
        }

        dialog.Update(++progress);
    }

    ModelSplatsHost modelHost(locations, shs, scales, opacities, rotations);

    progress += 1000;
    dialog.Update(progress);

    delete trainer->model;
    trainer->model = new ModelSplatsDevice(modelHost);
}

void UiFrame::onMenuButton(wxCommandEvent& event) {
    if (event.GetId() == FILE_NEW_PROJECT) {

        wxMessageDialog dialog(this, "This will reset the current project to a blank template! Are you sure?",
                               "Overwrite Project?", wxOK | wxCANCEL | wxCANCEL_DEFAULT | wxICON_EXCLAMATION);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        initProject();
        initFieldGrid();
        refreshProject();

    }if (event.GetId() == FILE_NEW_FIELD_GRID) {

        wxMessageDialog dialog(this, "This will reset the current splat field to a blank template! Are you sure?",
                               "Overwrite Splats?", wxOK | wxCANCEL | wxCANCEL_DEFAULT | wxICON_EXCLAMATION);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        initFieldGrid();
        refreshProject();

    }if (event.GetId() == FILE_NEW_FIELD_MONO) {

        wxMessageDialog dialog(this, "This will reset the current splat field to a blank template! Are you sure?",
                               "Overwrite Splats?", wxOK | wxCANCEL | wxCANCEL_DEFAULT | wxICON_EXCLAMATION);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        initFieldMono();
        refreshProject();

    }if (event.GetId() == FILE_NEW_FIELD_MODEL) {

        wxMessageDialog dialog(this, "This will reset the current splat field to a blank template! Are you sure?",
                               "Overwrite Splats?", wxOK | wxCANCEL | wxCANCEL_DEFAULT | wxICON_EXCLAMATION);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        initFieldModel();
        refreshProject();

    }else if (event.GetId() == FILE_SAVE_PROJECT) {

        wxDirDialog dialog(this, "Save Project", "");
        if (dialog.ShowModal() == wxID_CANCEL) return;
        saveSettings(dialog.GetPath().ToStdString() + "\\settings.json");
        saveSplats(dialog.GetPath().ToStdString() + "\\splats.gobj");

    }else if (event.GetId() == FILE_SAVE_SPLATS) {

        wxFileDialog dialog(this, "Save Gaussian Splats", "", "splats", "Gaussian OBJ Files (*.gobj)|*.gobj", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        saveSplats(dialog.GetPath().ToStdString());

    } else if (event.GetId() == FILE_SAVE_SETTINGS) {

        wxFileDialog dialog(this, "Save Settings", "", "settings", "JSON Files (*.json)|*.json", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        saveSettings(dialog.GetPath().ToStdString());

    } else if (event.GetId() == FILE_LOAD_PROJECT) {

        wxDirDialog dialog(this, "Load Project", "", wxDD_DIR_MUST_EXIST);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        loadSettings(dialog.GetPath().ToStdString() + "\\settings.json");
        loadSplats(dialog.GetPath().ToStdString() + "\\splats.gobj");
        refreshProject();

    } else if (event.GetId() == FILE_LOAD_SPLATS) {

        wxFileDialog dialog(this, "Load Gaussian Splats", "", "splats", "Gaussian OBJ Files (*.gobj)|*.gobj", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        loadSplats(dialog.GetPath().ToStdString());
        refreshProject();

    } else if (event.GetId() == FILE_LOAD_SETTINGS) {

        wxFileDialog dialog(this, "Load Settings", "", "settings", "JSON Files (*.json)|*.json", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        loadSettings(dialog.GetPath().ToStdString());
        refreshProject();

    } else if (event.GetId() == ABOUT_ABOUT) {
        UiDialogAbout dialog(this);
    }
}

void UiFrame::onPaint(wxPaintEvent& event) {
    if(!IsShown()) return;
    update();
}

void UiFrame::onIdle(wxIdleEvent& event) {
    update();
    event.RequestMore();
}

BEGIN_EVENT_TABLE(UiFrame, wxFrame)
EVT_PAINT(UiFrame::onPaint)
EVT_IDLE(UiFrame::onIdle)
END_EVENT_TABLE()
