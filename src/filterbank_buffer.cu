#include <algorithm>

#include "filterbank_buffer.cuh"

FilterbankBuffer::FilterbankBuffer(int gpuid) : gpuid_(gpuid) {
    cudaSetDevice(gpuid_);
    start_ = 0;
    end_ = 0;
}

FilterbankBuffer::FilterbankBuffer(int gulpno, size_t extrasize, size_t gulpsize, size_t totalsize, int gpuid) : extrasamples_(extrasize),
                                                                                gulpsamples_(gulpsize),
                                                                                nogulps_(gulpno),
                                                                                totalsamples_(totalsize),
                                                                                gpuid_(gpuid) {
    start_ = 0;
    end_ = 0;
    samplestate_ = new unsigned int[(int)totalsamples_];
    std::fill(samplestate_, samplestate_ + totalsamples_, 0);
}

FilterbankBuffer::~FilterbankBuffer() {
    end_ = 0;
}

void FilterbankBuffer::Allocate(int accumulate, int gulpno, size_t extrasize, size_t gulpsize, size_t totalsize, int filchans, int stokesno, int filbits) {
    fil_saved = 0;
    accumulate_ = accumulate;
    extrasamples_ = extrasize;
    gulpsamples_ = gulpsize;
    nochans_ = filchans;
    nogulps_ = gulpno;
    nostokes_ = stokesno;
    totalsamples_ = totalsize;
    typebytes_ = filbits / 8;

    gulptimes_ = new ObsTime[nogulps_];
    hdfilterbank = new unsigned char*[nostokes_];
    samplestate_ = new unsigned int[(int)totalsamples_];
    std::fill(samplestate_, samplestate_ + totalsamples_, 0);
    cudaCheckError(cudaHostAlloc((void**)&rambuffer_, nostokes_ * sizeof(unsigned char*), cudaHostAllocDefault));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&(hdfilterbank_[istoke]), totalsamples_ * nochans_ * typebytes_)));
        cudaCheckError(cudaHostAlloc((void**)&(rambuffer_[istoke]), (gulpsamples_ + extrasamples_) * nochans_ * 2 * typebytes_, cudaHostAllocDefault));
    }

    cudaCheckError(cudaMalloc((void**)&dfilterbank_, nostokes_ * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(dfilterbank_, hdfilterbank_, nostokes_ * sizeof(unsigned char*), cudaMemcpyHostToDevice));

}

void FilterbankBuffer::Deallocate(void) {

    cudaCheckError(cudaFree(dfilterbank_));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaFreeHost(rambuffer_[istoke]));
        cudaCheckError(cudaFree(hdfilterbank_[istoke]));
    }

    cudaCheckError(cudaFreeHost(rambuffer_));

    delete [] gulptimes_;
    delete [] hdfilterbank_;
    delete [] samplestate_;
}

void FilterbankBuffer::UpdateFilledTimes(ObsTime frame_time) {
    std::lock_guard<mutex> addguard(statemutex_);
    int framet = frame_time.framefromstart;
    int index = frame_time.framefromstart % (nogulps_ * gulpsamples_);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();

    for (int iacc = 0; iacc < accumulate_; iacc++) {
        index = framet % (nogulps_ * gulpsamples_);
        if((index % gulpsamples_) == 0)
            gulptimes_[index / gulpsamples_] = frame_time;
        samplestate_[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totalsamples_ << std::endl;
        //std::cout.flush();
        if ((index < extrasamples_) && (framet > extrasamples_)) {
            samplestate_[index + nogulps_ * gulpsamples_] = 1;
        }
        framet++;
    }
}

int FilterbankBuffer::CheckIfReady() {
    std::lock_guard<mutex> addguard(statemutex_);
    // for now check only the last position for the gulp
    for (int igulp = 0; igulp < nogulps_; igulp++) {
        if (samplestate_[(igulp + 1) * gulpsamples_ + extrasamples_ - 1] == 1)
            return (igulp + 1);
    }
    return 0;
}

ObsTime FilterbankBuffer::GetTime(int index) {
    return gulptimes_[index];
}

void FilterbankBuffer::SendToDisk(int idx, header_f header, std::string outdir) {
        SaveFilterbank(rambuffer_, gulpsamples_ + extrasamples_, (gulpsamples_ + extrasamples_) * nochans_ * nostokes_ * idx, header, nostokes_, fil_saved, outdir);
        fil_saved++;
}

void FilterbankBuffer::SendToRam(int idx, cudaStream_t &stream, int hostjump) {
    // NOTE: Which half of the RAM buffer we are saving into
    hostjump *= (gulpsamples_ + extrasamples_) * nochans_ * nostokes_;

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMemcpyAsyc(rambuffer_[istoke] + hostjump * typebytes_, hdfilterbank_[istoke] + (idx - 1) * gulpsamples_ * nochans_ * typebytes_, (gulpsamples_ + extrasamples_) * nochans_ * typebytes_, cudaMemcpyDeviceToHost, stream));
    }

    cudaStreamSynchronize(stream);
    statemutex_.lock();
    samplestate_[idx * gulpsamples_ + extrasamples_ - 1] = 0;
    statemutex_.unlock();
}
